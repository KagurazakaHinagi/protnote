"""Combine structure graphs + ESM-C embeddings into EGNN-ready .pt files.

Reads parsed structures and pre-computed ESM-C embeddings, builds atom-level
graphs with combined covalent + k-NN edges, and saves final tensors.

Usage:
    python bin/prepare_graph_data.py
    python bin/prepare_graph_data.py --knn-k 20
    python bin/prepare_graph_data.py --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from protnote.utils.configs import load_config
from protnote.utils.data import read_fasta
from protnote.utils.structure import (
    align_esmc_to_structure,
    build_protein_atom_graph,
    parse_structure,
    trim_terminal_tags,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_sequences_map(fasta_paths: list[Path]) -> dict[str, str]:
    """Build a mapping of sequence_id -> sequence from FASTA files."""
    seqs = {}
    for fasta_path in fasta_paths:
        if not fasta_path.exists():
            logger.warning(f"FASTA file not found: {fasta_path}")
            continue
        records = read_fasta(str(fasta_path))
        for sequence, seq_id, _ in records:
            if seq_id not in seqs:
                seqs[seq_id] = sequence
    return seqs


def main():
    parser = argparse.ArgumentParser(description="Prepare EGNN-ready graph data.")
    parser.add_argument(
        "--fasta-path-names",
        nargs="+",
        default=["TRAIN_DATA_PATH", "VAL_DATA_PATH", "TEST_DATA_PATH"],
        help="Config key names for FASTA files (used for sequence lookup).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=None,
        help="Number of k-NN neighbors. Defaults to config KNN_K.",
    )
    args = parser.parse_args()

    CONFIG, ROOT_PATH = load_config()
    DATA_PATH = ROOT_PATH / "data"

    knn_k = args.knn_k or CONFIG["params"].get("KNN_K", 20)

    # Load indices (paths already absolute from load_config)
    structure_dir = Path(CONFIG["paths"]["data_paths"].get("STRUCTURE_DIR", DATA_PATH / "structures"))
    structure_index_path = Path(
        CONFIG["paths"]["data_paths"].get("STRUCTURE_INDEX_PATH", DATA_PATH / "structures" / "structure_index.json")
    )
    esmc_dir = Path(CONFIG["paths"]["data_paths"].get("ESMC_EMBEDDING_DIR", DATA_PATH / "embeddings" / "esmc"))
    esmc_index_path = Path(CONFIG["paths"]["data_paths"].get("ESMC_INDEX_PATH", DATA_PATH / "embeddings" / "esmc" / "esmc_index.json"))

    if not structure_index_path.exists():
        logger.error(f"Structure index not found: {structure_index_path}. Run download_structures.py first.")
        return
    if not esmc_index_path.exists():
        logger.error(f"ESM-C index not found: {esmc_index_path}. Run generate_sequence_embeddings.py first.")
        return

    with open(structure_index_path) as f:
        structure_index = json.load(f)
    with open(esmc_index_path) as f:
        esmc_index = json.load(f)

    # Find proteins with both structure and ESM-C embeddings
    common_ids = sorted(set(structure_index.keys()) & set(esmc_index.keys()))
    logger.info(
        f"Found {len(common_ids)} proteins with both structure and ESM-C embeddings "
        f"(structure: {len(structure_index)}, ESM-C: {len(esmc_index)})"
    )

    if not common_ids:
        logger.error("No proteins with both structure and ESM-C data.")
        return

    # Load FASTA sequences for alignment (paths already absolute from load_config)
    fasta_paths = []
    for name in args.fasta_path_names:
        path = CONFIG["paths"]["data_paths"].get(name)
        if path:
            fasta_paths.append(path)
    sequence_map = collect_sequences_map(fasta_paths)

    # Setup output (paths already absolute from load_config)
    output_dir = Path(CONFIG["paths"]["data_paths"].get("PROCESSED_GRAPH_DIR", DATA_PATH / "processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_index_path = Path(CONFIG["paths"]["data_paths"].get("GRAPH_INDEX_PATH", DATA_PATH / "processed" / "graph_index.json"))

    # Load existing index
    if graph_index_path.exists():
        with open(graph_index_path) as f:
            graph_index = json.load(f)
    else:
        graph_index = {}

    # Filter to unprocessed
    remaining = [sid for sid in common_ids if sid not in graph_index]
    logger.info(f"{len(remaining)} proteins to process ({len(graph_index)} already done)")

    failed = []
    processed = 0

    for seq_id in tqdm(remaining, desc="Preparing graph data"):
        try:
            # Parse structure
            struct_info = structure_index[seq_id]
            cif_path = structure_dir / struct_info["path"]

            if not cif_path.exists():
                logger.warning(f"Structure file not found: {cif_path}")
                failed.append((seq_id, "structure_file_missing"))
                continue

            atom_array = parse_structure(cif_path)

            if atom_array.array_length() == 0:
                logger.warning(f"Empty structure for {seq_id}")
                failed.append((seq_id, "empty_structure"))
                continue

            # Trim expression tags / cloning artifacts from terminals
            fasta_seq = sequence_map.get(seq_id)
            if fasta_seq:
                atom_array, n_trim_n, n_trim_c = trim_terminal_tags(atom_array, fasta_seq)

            # Build atom graph
            graph = build_protein_atom_graph(atom_array, chains=struct_info["chain_ids"], k=knn_k)

            # Load ESM-C embeddings
            esmc_filename = esmc_index[seq_id]
            esmc_path = esmc_dir / esmc_filename
            esmc_emb = torch.load(esmc_path, weights_only=True)

            # Align ESM-C to structure
            fasta_seq = sequence_map.get(seq_id)
            aligned_emb = align_esmc_to_structure(
                esmc_emb,
                graph["residue_names"],
                fasta_seq=fasta_seq,
                structure_res_ids=graph["residue_res_ids"],
            )

            if aligned_emb is None:
                logger.warning(f"ESM-C alignment failed for {seq_id}")
                failed.append((seq_id, "alignment_failed"))
                continue

            # Build output dict
            output = {
                "sequence_id": seq_id,
                "sequence": fasta_seq or "",
                "coords": torch.tensor(graph["coords"], dtype=torch.float32),
                "atom_types": torch.tensor(graph["atom_types"], dtype=torch.long),
                "atom_names": graph["atom_names"],
                "residue_index": torch.tensor(graph["residue_index"], dtype=torch.long),
                "residue_names": graph["residue_names"],
                "residue_res_ids": torch.tensor(graph["residue_res_ids"], dtype=torch.long),
                "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                "edge_type": torch.tensor(graph["edge_type"], dtype=torch.long),
                "esmc_embeddings": aligned_emb,
                "n_atoms": graph["n_atoms"],
                "n_residues": graph["n_residues"],
                "structure_source": struct_info["source"],
            }

            # Save
            out_path = output_dir / f"{seq_id}.pt"
            torch.save(output, out_path)
            graph_index[seq_id] = f"{seq_id}.pt"
            processed += 1

        except Exception as e:
            logger.warning(f"Failed to process {seq_id}: {e}")
            failed.append((seq_id, str(e)))
            continue

        # Periodic save
        if processed % 500 == 0 and processed > 0:
            with open(graph_index_path, "w") as f:
                json.dump(graph_index, f, indent=2)

    # Final save
    with open(graph_index_path, "w") as f:
        json.dump(graph_index, f, indent=2)

    logger.info(f"Done. {len(graph_index)} total proteins processed, {len(failed)} failed.")

    if failed:
        failed_path = output_dir / "failed.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"Failed proteins listed in {failed_path}")

    # Print summary
    if graph_index:
        sample_id = next(iter(graph_index))
        sample = torch.load(output_dir / graph_index[sample_id], weights_only=False)
        logger.info(
            f"Sample output ({sample_id}): "
            f"atoms={sample['n_atoms']}, residues={sample['n_residues']}, "
            f"coords={list(sample['coords'].shape)}, "
            f"esmc={list(sample['esmc_embeddings'].shape)}, "
            f"edges={list(sample['edge_index'].shape)}"
        )


if __name__ == "__main__":
    main()
