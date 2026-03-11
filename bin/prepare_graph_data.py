"""Combine structure graphs + ESM-C embeddings into EGNN-ready .pt files.

Reads parsed structures and pre-computed ESM-C embeddings, builds atom-level
graphs with combined covalent + k-NN edges, and saves final tensors.

Supports multiprocessing for parallel graph construction and optional
consolidation into a single indexed archive file (.pngrph).

Example Usage:
    python bin/prepare_graph_data.py \
    --num-workers 8 \
    --knn-k 20 \
    --no-consolidate \
    --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import signal
from pathlib import Path

import torch
from tqdm import tqdm

from protnote.utils.configs import get_project_root, register_resolvers
from protnote.utils.data import read_fasta
from protnote.utils.graph_archive import (
    consolidate_to_archive,
    consolidate_to_sharded_archive,
    open_archive,
)
from protnote.utils.structure import extract_aa_residue_by_chain_ids

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
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


# -----------------------------------------------------------------------------
# Worker Process globals and functions for multiprocessing
# -----------------------------------------------------------------------------

_worker_ctx = {}


class _ProcessingTimeout(Exception):
    """Raised by SIGALRM when a single protein exceeds the time limit."""


def _sigalrm_handler(signum, frame):
    raise _ProcessingTimeout


def _worker_init(structure_dir, esmc_dir, structure_index, esmc_index, sequence_map, knn_k, local_afdb_dir=None, local_afdb_suffix=None, timeout=0):
    """Called once per worker process to store shared config and import heavy modules."""
    _worker_ctx["structure_dir"] = Path(structure_dir)
    _worker_ctx["esmc_dir"] = Path(esmc_dir)
    _worker_ctx["structure_index"] = structure_index
    _worker_ctx["esmc_index"] = esmc_index
    _worker_ctx["sequence_map"] = sequence_map
    _worker_ctx["knn_k"] = knn_k
    _worker_ctx["local_afdb_dir"] = Path(local_afdb_dir) if local_afdb_dir else None
    _worker_ctx["local_afdb_suffix"] = local_afdb_suffix
    _worker_ctx["timeout"] = timeout

    if timeout > 0:
        signal.signal(signal.SIGALRM, _sigalrm_handler)

    # Import heavy modules once per worker (avoid repeated import overhead)
    from protnote.utils.structure import (
        align_esmc_to_structure,
        build_protein_atom_graph,
        parse_structure,
        trim_terminal_tags,
    )

    _worker_ctx["parse_structure"] = parse_structure
    _worker_ctx["build_atom_graph"] = build_protein_atom_graph
    _worker_ctx["align_esmc_to_structure"] = align_esmc_to_structure
    _worker_ctx["trim_terminal_tags"] = trim_terminal_tags
    _worker_ctx["extract_aa_residue_by_chain_ids"] = extract_aa_residue_by_chain_ids


def _process_one(seq_id: str) -> dict:
    """Process a single protein. Returns result dict with status and optional data.

    Returns:
        {"seq_id": str, "status": "ok", "filename": str, "data_bytes": bytes}
        or {"seq_id": str, "status": "failed", "reason": str}
    """
    ctx = _worker_ctx
    timeout = ctx.get("timeout", 0)
    try:
        if timeout > 0:
            signal.alarm(timeout)

        struct_info = ctx["structure_index"][seq_id]

        # Resolve structure file path: use local AFDB folder for alphafolddb entries if configured
        if (
            ctx["local_afdb_dir"]
            and struct_info.get("source") == "alphafolddb"
        ):
            afdb_filename = f"AF-{seq_id}-F1-model_{ctx['local_afdb_suffix']}.pdb"
            cif_path = ctx["local_afdb_dir"] / afdb_filename
        else:
            cif_path = ctx["structure_dir"] / struct_info["path"]

        if not cif_path.exists():
            return {"seq_id": seq_id, "status": "failed", "reason": "structure_file_missing"}

        atom_array = ctx["parse_structure"](cif_path)

        # Filter out amino-acid Atoms with corresponding chain
        atom_array = ctx["extract_aa_residue_by_chain_ids"](atom_array, struct_info.get("chain_ids", ["A"]))

        if atom_array.array_length() == 0:
            return {"seq_id": seq_id, "status": "failed", "reason": "empty_structure"}

        # Trim expression tags / cloning artifacts from terminals
        fasta_seq = ctx["sequence_map"].get(seq_id)
        if fasta_seq:
            atom_array, n_trim_n, n_trim_c = ctx["trim_terminal_tags"](atom_array, fasta_seq)

        # Build atom graph
        graph = ctx["build_atom_graph"](atom_array, chains=struct_info.get("chain_ids", "A"), k=ctx["knn_k"])

        # Load ESM-C embeddings
        esmc_filename = ctx["esmc_index"][seq_id]
        esmc_path = ctx["esmc_dir"] / esmc_filename
        esmc_emb = torch.load(esmc_path, weights_only=True)

        # Align ESM-C to structure
        aligned_emb = ctx["align_esmc_to_structure"](
            esmc_emb,
            graph["residue_names"],
            fasta_seq=fasta_seq,
        )

        if aligned_emb is None:
            return {"seq_id": seq_id, "status": "failed", "reason": "alignment_failed"}

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

        # Serialize to bytes so the main process can write without re-serializing
        buf = io.BytesIO()
        torch.save(output, buf)

        return {
            "seq_id": seq_id,
            "status": "ok",
            "filename": f"{seq_id}.pt",
            "n_atoms": output["n_atoms"],
            "data_bytes": buf.getvalue(),
        }

    except _ProcessingTimeout:
        return {"seq_id": seq_id, "status": "failed", "reason": f"timeout (>{timeout}s)"}
    except Exception as e:
        return {"seq_id": seq_id, "status": "failed", "reason": str(e)}
    finally:
        if timeout > 0:
            signal.alarm(0)


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers. Use 1 for sequential (debug) mode.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10,
        help="Chunk size for multiprocessing imap_unordered.",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        default=False,
        help="Skip consolidation into a single archive file.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of archive shards (default: 1 = single file). "
        "Set to e.g. 16 for large datasets.",
    )
    parser.add_argument(
        "--keep-individual-files",
        action="store_true",
        default=False,
        help="Keep individual .pt files after archiving (default: delete them).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-protein timeout in seconds (default: 300). "
        "Proteins exceeding this are marked failed and retried on the next run. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--local-afdb",
        action="store_true",
        default=False,
        help="Use local AFDB folder for alphafolddb structures instead of downloaded CIF files. "
        "Expects files named AF-<UNIPROT_ID>-F1-model_<SUFFIX>.pdb in LOCAL_AFDB_DIR.",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Hydra paths config to use (e.g. 'mpnn_toxin'). Defaults to 'default'.",
    )
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    project_root = get_project_root()
    register_resolvers()
    GlobalHydra.instance().clear()
    overrides = [f"paths={args.paths}"] if args.paths else []
    with initialize_config_dir(version_base=None, config_dir=str(project_root / "configs")):
        cfg = compose(config_name="config", overrides=overrides)

    DATA_PATH = project_root / "data"

    knn_k = args.knn_k or cfg.params.get("KNN_K", 20)

    # Load indices (cfg has relative paths, prepend DATA_PATH)
    structure_dir = Path(DATA_PATH / cfg.paths.data_paths.get("STRUCTURE_DIR", "structures"))
    structure_index_path = Path(
        DATA_PATH / cfg.paths.data_paths.get("STRUCTURE_INDEX_PATH", "structures/structure_index.json")
    )
    esmc_dir = Path(DATA_PATH / cfg.paths.data_paths.get("ESMC_EMBEDDING_DIR", "embeddings/esmc"))
    esmc_index_path = Path(DATA_PATH / cfg.paths.data_paths.get("ESMC_INDEX_PATH", "embeddings/esmc/esmc_index.json"))

    # Local AFDB configuration
    local_afdb_dir = None
    local_afdb_suffix = None
    if args.local_afdb:
        local_afdb_rel = cfg.paths.data_paths.get("LOCAL_AFDB_DIR", "")
        if not local_afdb_rel:
            logger.error("--local-afdb requires LOCAL_AFDB_DIR to be set in config.")
            return
        local_afdb_dir = Path(DATA_PATH / local_afdb_rel)
        local_afdb_suffix = cfg.paths.data_paths.get("LOCAL_AFDB_SUFFIX", "v4")
        if not local_afdb_dir.exists():
            logger.error(f"Local AFDB directory not found: {local_afdb_dir}")
            return
        logger.info(f"Using local AFDB structures from {local_afdb_dir} (suffix={local_afdb_suffix})")

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

    # Load FASTA sequences for alignment (cfg has relative paths, prepend DATA_PATH)
    fasta_paths = []
    for name in args.fasta_path_names:
        rel_path = cfg.paths.data_paths.get(name)
        if rel_path:
            fasta_paths.append(Path(DATA_PATH / rel_path))
    sequence_map = collect_sequences_map(fasta_paths)

    # Setup output (cfg has relative paths, prepend DATA_PATH)
    output_dir = Path(DATA_PATH / cfg.paths.data_paths.get("PROCESSED_GRAPH_DIR", "processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_index_path = Path(DATA_PATH / cfg.paths.data_paths.get("GRAPH_INDEX_PATH", "processed/graph_index.json"))

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

    if remaining:
        failed = []
        processed = 0
        num_workers = args.num_workers

        if num_workers > 1 and len(remaining) > 1:
            # --- Parallel mode ---
            logger.info(f"Processing with {num_workers} workers (chunksize={args.chunksize})")
            with mp.Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(
                    str(structure_dir),
                    str(esmc_dir),
                    structure_index,
                    esmc_index,
                    sequence_map,
                    knn_k,
                    str(local_afdb_dir) if local_afdb_dir else None,
                    local_afdb_suffix,
                    args.timeout,
                ),
            ) as pool:
                results = pool.imap_unordered(_process_one, remaining, chunksize=args.chunksize)
                for result in tqdm(results, total=len(remaining), desc="Preparing graph data"):
                    if result["status"] == "ok":
                        out_path = output_dir / result["filename"]
                        with open(out_path, "wb") as f:
                            f.write(result["data_bytes"])
                        graph_index[result["seq_id"]] = {
                            "filename": result["filename"],
                            "n_atoms": result["n_atoms"],
                        }
                        processed += 1

                        if processed % 500 == 0:
                            with open(graph_index_path, "w") as f:
                                json.dump(graph_index, f, indent=2)
                    else:
                        logger.warning(f"Failed {result['seq_id']}: {result['reason']}")
                        failed.append((result["seq_id"], result["reason"]))
        else:
            # --- Sequential mode (num_workers=1 or single protein) ---
            logger.info("Processing sequentially (num-workers=1)")
            # Initialize worker context in main process
            _worker_init(
                str(structure_dir),
                str(esmc_dir),
                structure_index,
                esmc_index,
                sequence_map,
                knn_k,
                str(local_afdb_dir) if local_afdb_dir else None,
                local_afdb_suffix,
                args.timeout,
            )
            for seq_id in tqdm(remaining, desc="Preparing graph data"):
                result = _process_one(seq_id)
                if result["status"] == "ok":
                    out_path = output_dir / result["filename"]
                    with open(out_path, "wb") as f:
                        f.write(result["data_bytes"])
                    graph_index[result["seq_id"]] = {
                        "filename": result["filename"],
                        "n_atoms": result["n_atoms"],
                    }
                    processed += 1

                    if processed % 500 == 0:
                        with open(graph_index_path, "w") as f:
                            json.dump(graph_index, f, indent=2)
                else:
                    logger.warning(f"Failed {result['seq_id']}: {result['reason']}")
                    failed.append((result["seq_id"], result["reason"]))

        # Final save of graph index
        with open(graph_index_path, "w") as f:
            json.dump(graph_index, f, indent=2)

        logger.info(f"Done. {len(graph_index)} total proteins processed, {len(failed)} failed.")

        if failed:
            failed_path = output_dir / "failed.json"
            with open(failed_path, "w") as f:
                json.dump(failed, f, indent=2)
            logger.info(f"Failed proteins listed in {failed_path}")

    # --- Consolidation ---
    if not args.no_consolidate and graph_index:
        num_shards = args.num_shards
        if num_shards > 1:
            logger.info(f"Consolidating {len(graph_index)} graphs into {num_shards} shards in {output_dir}")
            n_archived = consolidate_to_sharded_archive(output_dir, graph_index, output_dir, num_shards)
        else:
            archive_path = output_dir / "graphs.pngrph"
            logger.info(f"Consolidating {len(graph_index)} graphs into {archive_path}")
            n_archived = consolidate_to_archive(output_dir, graph_index, archive_path)

        if n_archived > 0 and not args.keep_individual_files:
            logger.info("Removing individual .pt files...")
            removed = 0
            for entry in graph_index.values():
                pt_path = output_dir / entry["filename"]
                if pt_path.exists():
                    pt_path.unlink()
                    removed += 1
            logger.info(f"Removed {removed} individual .pt files.")

    # Print summary
    if graph_index:
        sample_id = next(iter(graph_index))
        # Try loading from archive (single or sharded), then individual file
        archive_path = output_dir / "graphs.pngrph"
        shard_files = list(output_dir.glob("graphs.shard-*-of-*.pngrph"))
        if archive_path.exists() or shard_files:
            archive_source = archive_path if archive_path.exists() else output_dir
            reader = open_archive(archive_source)
            sample = reader[sample_id]
            reader.close()
        else:
            sample = torch.load(output_dir / graph_index[sample_id]["filename"], weights_only=False)
        logger.info(
            f"Sample output ({sample_id}): "
            f"atoms={sample['n_atoms']}, residues={sample['n_residues']}, "
            f"coords={list(sample['coords'].shape)}, "
            f"esmc={list(sample['esmc_embeddings'].shape)}, "
            f"edges={list(sample['edge_index'].shape)}"
        )


if __name__ == "__main__":
    main()
