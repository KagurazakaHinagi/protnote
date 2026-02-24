"""
Structural dataset: same GO/label format as ProteinDataset but returns graph + PLM per sample.

Supports two data formats:
- Legacy: Precomputed graph+PLM pkl: dict[sequence_id] -> (x, plm, edge_index, edge_s)
- Atom-level: Graph archive (.pngrph) with ESM-C embeddings
"""

import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError:
    Data = None
    DataLoader = None

import blosum as bl

from protnote.utils.data import generate_vocabularies, get_vocab_mappings, read_fasta

# Amino acid to index mapping for vanilla mode
AA_TO_INDEX = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 1-indexed, 0=unknown


class ProteinStructureDataset(Dataset):
    """
    Dataset for structural encoder: each sample is (graph + PLM, labels).
    Labels and vocabulary match ProteinDataset (GO annotations).

    Supports two data formats:
    - Legacy: graph_plm_pkl_path -> pkl mapping sequence_id -> (x, plm, edge_index, edge_s)
    - Atom-level: graph_dir + graph_index + optional graph_archive_path
    """

    def __init__(
        self,
        data_paths: dict,
        config: dict,
        logger=None,
        label_tokenizer=None,
        graph_plm_pkl_path: str = None,
        # New params for atom-level format
        graph_dir: str = None,
        graph_index: dict = None,
        graph_archive_path: str = None,
        use_atom_level: bool = False,
        split: str = None,
    ):
        """
        data_paths: same as ProteinDataset (data_path, dataset_type, annotations_path, vocabularies_dir).

        For legacy format:
            graph_plm_pkl_path: path to pkl mapping sequence_id -> (x, plm, edge_index, edge_s).

        For atom-level format:
            graph_dir: Directory containing .pt graph files
            graph_index: Dict mapping sequence_id -> filename
            graph_archive_path: Optional path to .pngrph archive
            use_atom_level: Set True to use atom-level format
            split: Dataset split (train/validation/test), used for logging
        """
        self.logger = logger or logging.getLogger(__name__)
        required_keys = ["data_path", "dataset_type"]
        for key in required_keys:
            if key not in data_paths:
                raise ValueError(f"Missing required key in data_paths: {key}")

        self.dataset_type = data_paths["dataset_type"]
        self.data_path = data_paths["data_path"]
        self.use_atom_level = use_atom_level
        self.label_tokenizer = label_tokenizer
        self.split = split or self.dataset_type

        self.augment_residue_probability = config["params"].get("AUGMENT_RESIDUE_PROBABILITY", 0.0)
        self.label_augmentation_descriptions = config["params"].get("LABEL_AUGMENTATION_DESCRIPTIONS", "name+label").split("+")
        self.inference_go_descriptions = config["params"].get("INFERENCE_GO_DESCRIPTIONS", "name+label").split("+")

        blosum62 = bl.BLOSUM(62)
        self.blosum62 = defaultdict(
            dict,
            {aa1: {aa2: blosum62[aa1][aa2] for aa2 in blosum62.keys()} for aa1 in blosum62.keys()},
        )

        # Initialize based on format
        if use_atom_level:
            self._init_atom_level(data_paths, config, graph_dir, graph_index, graph_archive_path)
        else:
            self._init_legacy(data_paths, config, graph_plm_pkl_path)

    def _init_legacy(self, data_paths, config, graph_plm_pkl_path):
        """Initialize for legacy pkl format."""
        if graph_plm_pkl_path is None:
            graph_plm_pkl_path = data_paths.get("graph_plm_pkl_path")
        if not graph_plm_pkl_path:
            raise ValueError("ProteinStructureDataset requires graph_plm_pkl_path for legacy format.")

        self.graph_plm_pkl_path = graph_plm_pkl_path

        # Load graph+PLM pkl and restrict to sequences that have structure
        with open(graph_plm_pkl_path, "rb") as f:
            self.graph_plm_pkl = pickle.load(f)
        self.valid_ids = set(self.graph_plm_pkl.keys())
        self._finish_init(data_paths, config)

    def _init_atom_level(self, data_paths, config, graph_dir, graph_index, graph_archive_path):
        """Initialize for atom-level graph archive format."""
        # Get paths from data_paths if not provided directly
        if graph_dir is None:
            graph_dir = data_paths.get("graph_dir") or config["paths"].get("GRAPH_DIR")
        if graph_index is None:
            graph_index_path = data_paths.get("graph_index_path") or config["paths"].get("GRAPH_INDEX_PATH")
            if graph_index_path and os.path.exists(graph_index_path):
                with open(graph_index_path) as f:
                    graph_index = json.load(f)
            else:
                graph_index = {}
        if graph_archive_path is None:
            graph_archive_path = data_paths.get("graph_archive_path") or config["paths"].get("GRAPH_ARCHIVE_PATH")

        self.graph_dir = graph_dir
        self.graph_index = graph_index
        self.graph_archive_path = graph_archive_path
        self._archive_reader = None  # Lazy init for DataLoader spawn safety

        self.valid_ids = set(graph_index.keys())
        self._finish_init(data_paths, config)

    def _finish_init(self, data_paths, config):
        # Load FASTA and filter to valid sequence_ids
        raw_data = read_fasta(data_paths["data_path"])
        self.data = [(seq, sid, labels) for (seq, sid, labels) in raw_data if sid in self.valid_ids]
        mode_str = "atom-level" if self.use_atom_level else "legacy"
        self.logger.info(
            f"ProteinStructureDataset [{self.split}] ({mode_str}): {len(self.data)} samples with structure (from {len(raw_data)} in FASTA)."
        )

        subset_fraction = config["params"].get(f"{self.dataset_type.upper()}_SUBSET_FRACTION", 1.0)
        if subset_fraction < 1.0:
            self.data = self.data[: int(subset_fraction * len(self.data))]

        extract_vocabularies_from = config["params"].get("EXTRACT_VOCABULARIES_FROM")
        vocabulary_path = config["paths"][extract_vocabularies_from] if extract_vocabularies_from else self.data_path
        self._preprocess_data(
            deduplicate=config["params"].get("DEDUPLICATE", True),
            max_sequence_length=config["params"].get("MAX_SEQUENCE_LENGTH"),
            vocabulary_path=vocabulary_path,
        )

        # Label embeddings (same as ProteinDataset)
        INDEX_OUTPUT_PATH = config["LABEL_EMBEDDING_PATH"].rsplit(".", 1)
        INDEX_OUTPUT_PATH = "_".join([INDEX_OUTPUT_PATH[0], "index"]) + "." + INDEX_OUTPUT_PATH[1]
        index_mapping = torch.load(INDEX_OUTPUT_PATH, weights_only=False)
        emb_tensor = torch.load(config["LABEL_EMBEDDING_PATH"], weights_only=False)
        (
            self.label_embeddings_index,
            self.label_embeddings,
            self.label_token_counts,
            self.label_descriptions,
        ) = self._process_label_embedding_mapping(mapping=index_mapping, embeddings=emb_tensor)
        self.sorted_label_embeddings, self.sorted_label_token_counts = self._sort_label_embeddings()

    def _get_archive_reader(self):
        """Lazily initialize the archive reader on first access."""
        if self._archive_reader is None and hasattr(self, "graph_archive_path") and self.graph_archive_path:
            from protnote.utils.graph_archive import open_archive

            self._archive_reader = open_archive(self.graph_archive_path)
        return self._archive_reader

    def _load_graph_atom_level(self, sequence_id, sequence):
        """Load atom-level graph data for a protein.

        Returns a dict with atom_coords, atom_types, atom_to_residue,
        esmc_embeddings, edge_index, num_residues, num_atoms.
        """
        # Try archive first
        reader = self._get_archive_reader()
        if reader is not None and sequence_id in reader:
            try:
                return reader[sequence_id]
            except (KeyError, EOFError, OSError, RuntimeError) as e:
                self.logger.warning(f"Failed to load graph for {sequence_id} from archive: {e}. Trying individual file.")

        # Then try individual file
        if sequence_id in self.graph_index and self.graph_dir is not None:
            graph_path = os.path.join(self.graph_dir, self.graph_index[sequence_id]["filename"])
            try:
                graph_data = torch.load(graph_path, weights_only=False)
                return graph_data
            except (FileNotFoundError, EOFError, OSError, RuntimeError) as e:
                self.logger.warning(f"Failed to load graph for {sequence_id}: {e}. Using fallback.")

        # Fallback: one virtual atom per residue, no edges
        num_residues = len(sequence)
        graph_data = {
            "coords": torch.zeros(num_residues, 3),
            "atom_types": torch.zeros(num_residues, dtype=torch.long),
            "residue_index": torch.arange(num_residues),
            "esmc_embeddings": torch.zeros(num_residues, 960),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "n_residues": num_residues,
            "n_atoms": num_residues,
        }
        return graph_data

    def _preprocess_data(self, deduplicate, max_sequence_length, vocabulary_path):
        self.logger.info("Cleaning structural data...")
        df = pd.DataFrame(self.data, columns=["sequence", "sequence_id", "labels"])
        if deduplicate:
            df = df.drop_duplicates(subset="sequence", keep="first")
        if max_sequence_length is not None and self.dataset_type == "train":
            df = df[df["sequence"].apply(len) <= max_sequence_length]
        self.data = list(df.itertuples(index=False, name=None))

        from collections import Counter

        label_freq = Counter()
        for _, _, labels in self.data:
            for lab in labels:
                label_freq[lab] += 1
        self.label_frequency = label_freq

        if vocabulary_path != self.data_path:
            vocabularies = generate_vocabularies(file_path=vocabulary_path)
        else:
            vocabularies = generate_vocabularies(data=self.data)
        self.amino_acid_vocabulary = vocabularies["amino_acid_vocab"]
        self.label_vocabulary = vocabularies["label_vocab"]
        self.sequence_id_vocabulary = vocabularies["sequence_id_vocab"]
        self.represented_vocabulary_mask = [label in self.label_frequency for label in self.label_vocabulary]
        self._process_amino_acid_vocab()
        self._process_label_vocab()
        self._process_sequence_id_vocab()

    def _process_amino_acid_vocab(self):
        self.aminoacid2int, self.int2aminoacid = get_vocab_mappings(self.amino_acid_vocabulary)

    def _process_label_vocab(self):
        self.label2int, self.int2label = get_vocab_mappings(self.label_vocabulary)

    def _process_sequence_id_vocab(self):
        self.sequence_id2int, self.int2sequence_id = get_vocab_mappings(self.sequence_id_vocabulary)

    def _process_label_embedding_mapping(self, mapping, embeddings):
        import pandas as pd

        assert set(self.inference_go_descriptions).issubset(
            set(["name", "label"])
        ), "only supporting name, label or name+label"

        if not isinstance(mapping, pd.DataFrame):
            mapping = pd.DataFrame(mapping)
        descriptions_considered = self.label_augmentation_descriptions if self.dataset_type == "train" else self.inference_go_descriptions
        mask = (mapping["description_type"].isin(descriptions_considered)) & (mapping["id"].isin(self.label_vocabulary))
        if hasattr(mask, "values"):
            mask = mask.values
        mapping = mapping.loc[mask].reset_index(drop=True).reset_index()
        embeddings = embeddings[mask]
        token_counts = mapping["token_count"].values
        mapping = mapping.groupby("id").agg(min_idx=("index", "min"), max_idx=("index", "max")).to_dict(orient="index")
        return mapping, embeddings, token_counts, None

    def _sort_label_embeddings(self):
        idx_list = []
        for go_term in self.label_vocabulary:
            idx_list.extend(
                range(
                    self.label_embeddings_index[go_term]["min_idx"],
                    self.label_embeddings_index[go_term]["max_idx"] + 1,
                )
            )
        return (
            self.label_embeddings[idx_list],
            self.label_token_counts[idx_list],
        )

    def _sample_label_embeddings(self):
        """Sample one embedding per label (for training with multiple description types)."""
        label_embedding_idxs_list = []
        for go_term in self.label_vocabulary:
            idx = np.random.randint(
                low=self.label_embeddings_index[go_term]["min_idx"],
                high=self.label_embeddings_index[go_term]["max_idx"] + 1,
            )
            label_embedding_idxs_list.append(idx)
        return (
            self.label_embeddings[label_embedding_idxs_list],
            self.label_token_counts[label_embedding_idxs_list],
        )

    def get_atom_counts(self):
        """Return an array of atom counts aligned with dataset indices."""
        counts = np.empty(len(self.data), dtype=np.int64)
        for i, (_sequence, seq_id, _labels) in enumerate(self.data):
            counts[i] = self.graph_index[seq_id]["n_atoms"]
        return counts

    def __len__(self):
        return len(self.data)

    def calculate_label_weights(self, inv_freq=True, normalize=True, return_list=True, power=0.5):
        """Same interface as ProteinDataset.calculate_label_weights for weighted sampling/loss."""
        import numpy as np

        counts = np.array([self.label_frequency.get(lab, 0) for lab in self.label_vocabulary], dtype=np.float64)
        counts = np.clip(counts, 1e-6, None)
        if inv_freq:
            weights = 1.0 / (counts**power)
        else:
            weights = counts
        if normalize:
            num_labels = len(self.label_vocabulary)
            weights = weights * num_labels / weights.sum()
        if return_list:
            return torch.tensor(weights, dtype=torch.float32)
        # Return dict mapping label names to weights (same as ProteinDataset)
        return {label: weights[i] for i, label in enumerate(self.label_vocabulary)}

    def calculate_pos_weight(self):
        """Same interface as ProteinDataset.calculate_pos_weight for BCE."""
        num_pos = sum(self.label_frequency.get(lab, 0) for lab in self.label_vocabulary)
        num_neg = len(self.data) * len(self.label_vocabulary) - num_pos
        return torch.tensor(num_neg / max(num_pos, 1e-6), dtype=torch.float32)

    def __getitem__(self, idx):
        sequence, sequence_id, labels = self.data[idx]

        labels_ints = [self.label2int[label] for label in labels]
        label_multihots = torch.nn.functional.one_hot(
            torch.tensor(labels_ints, dtype=torch.long),
            num_classes=len(self.label_vocabulary),
        ).sum(dim=0)

        # During training with multiple augmentation descriptions, sample one embedding per label
        if self.dataset_type == "train" and len(self.label_augmentation_descriptions) > 1:
            label_embeddings, label_token_counts = self._sample_label_embeddings()
            if not isinstance(label_token_counts, torch.Tensor):
                label_token_counts = torch.tensor(label_token_counts)
        else:
            label_embeddings = self.sorted_label_embeddings
            label_token_counts = torch.tensor(self.sorted_label_token_counts) if not isinstance(self.sorted_label_token_counts, torch.Tensor) else self.sorted_label_token_counts

        if self.use_atom_level:
            return self._getitem_atom_level(sequence, sequence_id, label_multihots, label_embeddings, label_token_counts)
        else:
            return self._getitem_legacy(sequence_id, label_multihots, label_embeddings, label_token_counts)

    def _getitem_legacy(self, sequence_id, label_multihots, label_embeddings, label_token_counts):
        """Get item for legacy pkl format (residue-level, PyG Data)."""
        x, plm, edge_index, edge_s = self.graph_plm_pkl[sequence_id]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if not isinstance(plm, torch.Tensor):
            plm = torch.as_tensor(plm, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if not isinstance(edge_s, torch.Tensor):
            edge_s = torch.as_tensor(edge_s, dtype=torch.float32)
        if edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)

        graph_data = Data(x=x, plm=plm, edge_index=edge_index, edge_s=edge_s)

        return {
            "structure_batch": graph_data,
            "sequence_id": sequence_id,
            "label_multihots": label_multihots,
            "label_embeddings": label_embeddings,
            "label_token_counts": label_token_counts,
        }

    def _getitem_atom_level(self, sequence, sequence_id, label_multihots, label_embeddings, label_token_counts):
        """Get item for atom-level format (ESM-C + atom types)."""
        graph_data = self._load_graph_atom_level(sequence_id, sequence)

        # Extract fields with consistent naming
        atom_coords = graph_data.get("coords", graph_data.get("atom_coords"))
        atom_types = graph_data["atom_types"]
        atom_to_residue = graph_data.get("residue_index", graph_data.get("atom_to_residue"))
        esmc_embeddings = graph_data["esmc_embeddings"]
        edge_index = graph_data["edge_index"]
        num_residues = graph_data.get("n_residues", graph_data.get("num_residues"))
        num_atoms = graph_data.get("n_atoms", graph_data.get("num_atoms"))

        # Build per-atom amino acid indices for vanilla mode
        per_residue_aa = torch.tensor([AA_TO_INDEX.get(aa, 0) for aa in sequence], dtype=torch.long)
        clamped_res = atom_to_residue.clamp(max=len(sequence) - 1)
        residue_indices = per_residue_aa[clamped_res]

        return {
            "atom_coords": atom_coords,
            "atom_types": atom_types,
            "atom_to_residue": atom_to_residue,
            "esmc_embeddings": esmc_embeddings,
            "edge_index": edge_index,
            "num_residues": num_residues,
            "num_atoms": num_atoms,
            "residue_indices": residue_indices,
            "sequence_id": sequence_id,
            "sequence_str": sequence,
            "label_multihots": label_multihots,
            "label_embeddings": label_embeddings,
            "label_token_counts": label_token_counts,
        }


def create_structural_loaders(
    datasets: dict,
    params: dict,
    label_sample_sizes: dict = None,
    shuffle_labels: bool = False,
    in_batch_sampling: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0,
    sequence_weights=None,
):
    """Create DataLoaders for structural encoder; uses collate_structure_batch."""
    from collections import defaultdict
    from functools import partial

    from torch.utils.data import DataLoader

    from protnote.data.collators import collate_structure_batch
    from protnote.data.samplers import DynamicBatchSampler, observation_sampler_factory

    max_atoms = params.get("MAX_ATOMS_PER_BATCH")

    loaders = defaultdict(list)
    for dataset_type, dataset_list in datasets.items():
        batch_size_for_type = params[f"{dataset_type.upper()}_BATCH_SIZE"]
        label_sample_size = label_sample_sizes.get(dataset_type) if label_sample_sizes else None
        for dataset in dataset_list:
            sequence_sampler = observation_sampler_factory(
                distribute_labels=params.get("DISTRIBUTE_LABELS", False),
                weighted_sampling=params.get("WEIGHTED_SAMPLING", False) and (dataset_type == "train"),
                dataset=dataset,
                world_size=world_size,
                rank=rank,
                sequence_weights=sequence_weights,
                shuffle=(dataset_type == "train"),
            )
            collate_fn = partial(
                collate_structure_batch,
                label_sample_size=label_sample_size,
                distribute_labels=params.get("DISTRIBUTE_LABELS", False),
                shuffle_labels=shuffle_labels,
                in_batch_sampling=in_batch_sampling and (dataset_type == "train"),
                world_size=world_size,
                rank=rank,
            )

            use_dynamic = (
                max_atoms is not None
                and hasattr(dataset, "get_atom_counts")
                and getattr(dataset, "use_atom_level", False)
            )
            if use_dynamic:
                atom_counts = dataset.get_atom_counts()
                batch_sampler = DynamicBatchSampler(
                    element_sampler=sequence_sampler,
                    atom_counts=atom_counts,
                    max_atoms_per_batch=max_atoms,
                    drop_last=(dataset_type == "train"),
                )
                loader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size_for_type,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=(dataset_type == "train"),
                    sampler=sequence_sampler,
                )
            loaders[dataset_type].append(loader)
    return loaders
