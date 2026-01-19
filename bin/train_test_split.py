"""Separate filtered SwissProt FASTA files into train/validation/test sets"""

import argparse
import logging
import os

from Bio import SeqIO
from sklearn.model_selection import train_test_split


def split_fasta_clustered(
    input_fasta: os.PathLike, output_dir: os.PathLike, train_val_test_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
):
    raise NotImplementedError("split_fasta_clustered is not implemented yet.")


def split_fasta_random(
    input_fasta: os.PathLike, output_dir: os.PathLike, train_val_test_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
):
    """Reads a FASTA file and splits it into train/validation/test sets randomly.

    Args:
        input_fasta (os.PathLike): Path to the input FASTA file.
        output_dir (os.PathLike): Path to the output directory.
        train_val_test_ratio (tuple[float, float, float], optional): Ratio of train/validation/test sets. Defaults to (0.8, 0.1, 0.1).
        seed (int, optional): Random seed. Defaults to 42.
    """
    assert sum(train_val_test_ratio) == 1.0, "The sum of train_val_test_ratio must be 1.0"
    assert os.path.exists(input_fasta), f"Input FASTA file {input_fasta} does not exist"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the FASTA file
    seqs = list(SeqIO.parse(input_fasta, "fasta"))
    logging.info(f"Read {len(seqs)} records from {input_fasta}")

    # Split the records into train/validation/test sets
    train_seqs, test_seqs = train_test_split(seqs, test_size=train_val_test_ratio[2], random_state=seed)
    train_seqs, val_seqs = train_test_split(
        train_seqs, test_size=train_val_test_ratio[1] / (1 - train_val_test_ratio[2]), random_state=seed
    )

    # Write the train/validation/test sets to separate FASTA files
    SeqIO.write(train_seqs, os.path.join(output_dir, "train.fasta"), "fasta")
    SeqIO.write(val_seqs, os.path.join(output_dir, "val.fasta"), "fasta")
    SeqIO.write(test_seqs, os.path.join(output_dir, "test.fasta"), "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a FASTA file into train/validation/test sets")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--train-val-test-ratio", type=float, nargs=3, default=(0.8, 0.1, 0.1), help="Ratio of train/validation/test sets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="random", help="Splitting mode")
    args = parser.parse_args()

    assert args.mode in ["random", "clustered"], f"Invalid mode: {args.mode}"

    if args.mode == "random":
        split_fasta_random(
            input_fasta=args.input_fasta, output_dir=args.output_dir, train_val_test_ratio=args.train_val_test_ratio, seed=args.seed
        )
    elif args.mode == "clustered":
        split_fasta_clustered(
            input_fasta=args.input_fasta, output_dir=args.output_dir, train_val_test_ratio=args.train_val_test_ratio, seed=args.seed
        )
