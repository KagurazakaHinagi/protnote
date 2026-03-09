"""Run LigandMPNN on toxin proteins with AlphaFold DB structures.

Reads a SwissProt pickle DataFrame, locates corresponding local AFDB
structure files, and runs LigandMPNN (bin/mpnn/run.py) on each via subprocess.

Usage:
    pixi run -e mpnn python bin/run_ligandmpnn.py \
        --number_of_batches 10 \
        --out_folder outputs/mpnn_toxin
"""

import argparse
import gzip
import logging
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/paths/default.yaml")
MPNN_SCRIPT = Path("bin/mpnn/run.py")



def resolve_afdb_pdb(afdb_dir: Path, suffix: str, uniprot_id: str) -> Path | None:
    """Find local AFDB PDB, decompressing .pdb.gz in-place if needed.

    Returns the .pdb path if found (or decompressed), None otherwise.
    """
    stem = f"AF-{uniprot_id}-F1-model_{suffix}"
    pdb_path = afdb_dir / f"{stem}.pdb"
    if pdb_path.exists():
        return pdb_path
    gz_path = afdb_dir / f"{stem}.pdb.gz"
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f_in, open(pdb_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        return pdb_path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder for LigandMPNN results")
    parser.add_argument("--number_of_batches", type=int, default=1, help="Number of batches for LigandMPNN")
    parser.add_argument("--pickle_path", type=str, default=None, help="Path to the toxin pickle file (default: from config)")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--seed", type=int, default=111, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cfg = OmegaConf.load(CONFIG_PATH)
    paths = {k: data_dir / v for k, v in cfg.data_paths.items()}
    local_afdb_dir = paths["LOCAL_AFDB_DIR"]
    local_afdb_suffix = cfg.data_paths.LOCAL_AFDB_SUFFIX
    checkpoint_path = Path(cfg.data_paths.LIGAND_MPNN_CHECKPOINT)
    pickle_path = args.pickle_path or str(paths["SWISSPROT_PICKLE_2025_01_PATH"])

    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    # Deduplicate by sequence, keeping the first occurrence
    n_before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first")
    log.info(f"Deduplicated: {n_before} -> {len(df)} proteins ({n_before - len(df)} duplicate sequences removed)")

    # Filter to proteins with AFDB structure IDs
    df_with_struct = df[df["struct_afdb"].notna()]
    log.info(f"With AFDB structure ID: {len(df_with_struct)}")

    # Check which ones have local PDB files (decompress .pdb.gz if needed)
    pdb_tasks = []
    missing = 0
    for _, row in df_with_struct.iterrows():
        uid = row["struct_afdb"]
        pdb_path = resolve_afdb_pdb(local_afdb_dir, local_afdb_suffix, uid)
        if pdb_path is not None:
            pdb_tasks.append((uid, pdb_path))
        else:
            missing += 1

    log.info(f"Found {len(pdb_tasks)} local PDB files, {missing} missing")

    if not pdb_tasks:
        log.error(f"No structure files found in {local_afdb_dir}. Download them first.")
        sys.exit(1)

    os.makedirs(args.out_folder, exist_ok=True)

    failed = []
    for i, (uid, pdb_path) in enumerate(pdb_tasks):
        log.info(f"[{i + 1}/{len(pdb_tasks)}] Running LigandMPNN on {uid}")
        cmd = [
            sys.executable,
            str(MPNN_SCRIPT),
            "--model_type",
            "ligand_mpnn",
            "--checkpoint_ligand_mpnn",
            str(checkpoint_path),
            "--seed",
            str(args.seed),
            "--pdb_path",
            str(pdb_path),
            "--out_folder",
            args.out_folder,
            "--batch_size",
            "1",
            "--number_of_batches",
            str(args.number_of_batches),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning(f"Failed on {uid}: {result.stderr.strip()}")
            failed.append(uid)
        else:
            log.info(f"Done {uid}")

    log.info(f"Completed: {len(pdb_tasks) - len(failed)}/{len(pdb_tasks)}")
    if failed:
        log.warning(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
