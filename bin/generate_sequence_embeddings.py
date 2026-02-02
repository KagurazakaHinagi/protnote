"""Generate per-residue ESM-C embeddings for all proteins.

Reads FASTA files, processes sequences through ESM-C 300m in batches,
and saves per-residue embeddings as individual .pt files.

Usage:
    python bin/generate_sequence_embeddings.py
    python bin/generate_sequence_embeddings.py --batch-size 16 --model-name esmc_300m
    python bin/generate_sequence_embeddings.py --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from protnote.utils.configs import load_config
from protnote.utils.data import read_fasta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_sequences(fasta_paths: list[Path]) -> list[tuple[str, str]]:
    """Extract unique (sequence_id, sequence) pairs from FASTA files."""
    seen = {}
    for fasta_path in fasta_paths:
        if not fasta_path.exists():
            logger.warning(f"FASTA file not found: {fasta_path}")
            continue
        records = read_fasta(str(fasta_path))
        for sequence, seq_id, _ in records:
            if seq_id not in seen:
                seen[seq_id] = sequence
    pairs = sorted(seen.items())
    logger.info(f"Collected {len(pairs)} unique sequences from {len(fasta_paths)} FASTA files")
    return pairs


def group_by_length(pairs: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    """Group sequences by similar length to minimize padding waste."""
    sorted_pairs = sorted(pairs, key=lambda x: len(x[1]))
    batches = []
    for i in range(0, len(sorted_pairs), batch_size):
        batches.append(sorted_pairs[i : i + batch_size])
    return batches


def main():
    parser = argparse.ArgumentParser(description="Generate per-residue ESM-C embeddings.")
    parser.add_argument(
        "--fasta-path-names",
        nargs="+",
        default=["TRAIN_DATA_PATH", "VAL_DATA_PATH", "TEST_DATA_PATH"],
        help="Config key names for FASTA files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="ESM-C model name. Defaults to config ESMC_MODEL_NAME.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for ESM-C inference. Defaults to config ESMC_BATCH_SIZE.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=5000,
        help="Skip sequences longer than this.",
    )
    args = parser.parse_args()

    CONFIG, ROOT_PATH = load_config()
    DATA_PATH = ROOT_PATH / "data"

    model_name = args.model_name or CONFIG["params"].get("ESMC_MODEL_NAME", "esmc_300m")
    batch_size = args.batch_size or CONFIG["params"].get("ESMC_BATCH_SIZE", 8)
    embedding_dim = CONFIG["params"].get("ESMC_EMBEDDING_DIM", 960)

    # Resolve FASTA paths
    fasta_paths = []
    for name in args.fasta_path_names:
        path = CONFIG["paths"]["data_paths"].get(name)
        if path:
            fasta_paths.append(path)

    # Collect sequences
    all_pairs = collect_sequences(fasta_paths)

    # Setup output directory
    emb_dir = Path(CONFIG["paths"]["data_paths"].get("ESMC_EMBEDDING_DIR", DATA_PATH / "embeddings" / "esmc"))
    emb_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(CONFIG["paths"]["data_paths"].get("ESMC_INDEX_PATH", DATA_PATH / "embeddings" / "esmc" / "esmc_index.json"))

    # Load existing index
    if index_path.exists():
        with open(index_path) as f:
            esmc_index = json.load(f)
        logger.info(f"Loaded existing index with {len(esmc_index)} entries")
    else:
        esmc_index = {}

    # Filter to unprocessed sequences
    remaining = [(sid, seq) for sid, seq in all_pairs if sid not in esmc_index]
    logger.info(f"{len(remaining)} sequences to process ({len(esmc_index)} already done)")

    if not remaining:
        logger.info("All sequences already processed.")
        return

    # Filter out sequences that are too long
    skipped = [(sid, seq) for sid, seq in remaining if len(seq) > args.max_sequence_length]
    remaining = [(sid, seq) for sid, seq in remaining if len(seq) <= args.max_sequence_length]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} sequences longer than {args.max_sequence_length}")

    # Load ESM-C model
    logger.info(f"Loading ESM-C model: {model_name}")
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMCInferenceClient

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: ESMCInferenceClient = ESMC.from_pretrained(model_name, device=device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Process in batches
    batches = group_by_length(remaining, batch_size)
    processed = 0

    for batch in tqdm(batches, desc="Generating ESM-C embeddings"):
        seq_ids = [sid for sid, _ in batch]
        sequences = [seq for _, seq in batch]

        try:
            with torch.no_grad():
                # Tokenize batch
                tokens = model._tokenize(sequences)

                # Forward pass
                output = model.forward(sequence_tokens=tokens)

                # Extract per-residue embeddings (remove BOS/EOS tokens)
                embeddings = output.embeddings  # [B, L_padded, D]

                for i, (sid, seq) in enumerate(zip(seq_ids, sequences)):
                    seq_len = len(seq)
                    # Slice: skip BOS token (index 0), take seq_len tokens, skip EOS
                    emb = embeddings[i, 1 : 1 + seq_len, :].cpu().to(torch.float32)

                    if emb.shape != (seq_len, embedding_dim):
                        logger.warning(
                            f"Unexpected embedding shape for {sid}: got {emb.shape}, expected ({seq_len}, {embedding_dim}). Skipping."
                        )
                        continue

                    # Save
                    out_path = emb_dir / f"{sid}.pt"
                    torch.save(emb, out_path)
                    esmc_index[sid] = f"{sid}.pt"
                    processed += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"OOM on batch with max length {max(len(s) for s in sequences)}. Falling back to single-sequence processing."
                )
                torch.cuda.empty_cache()

                # Process one at a time
                for sid, seq in zip(seq_ids, sequences):
                    try:
                        with torch.no_grad():
                            tokens = model._tokenize([seq])
                            output = model.forward(sequence_tokens=tokens)
                            emb = output.embeddings[0, 1 : 1 + len(seq), :].cpu().to(torch.float32)

                            if emb.shape[0] == len(seq) and emb.shape[1] == embedding_dim:
                                out_path = emb_dir / f"{sid}.pt"
                                torch.save(emb, out_path)
                                esmc_index[sid] = f"{sid}.pt"
                                processed += 1
                    except RuntimeError:
                        logger.warning(f"Failed to process {sid} (length {len(seq)}). Skipping.")
                        torch.cuda.empty_cache()
            else:
                raise

        # Periodic save
        if processed % 1000 == 0 and processed > 0:
            with open(index_path, "w") as f:
                json.dump(esmc_index, f, indent=2)

    # Final save
    with open(index_path, "w") as f:
        json.dump(esmc_index, f, indent=2)

    logger.info(f"Done. {len(esmc_index)} total embeddings saved to {emb_dir}")
    if skipped:
        logger.info(f"{len(skipped)} sequences skipped due to length > {args.max_sequence_length}")


if __name__ == "__main__":
    main()
