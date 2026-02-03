"""Download protein structure from (PDB/AlphaFoldDB) via UniProt cross-references.

Map UniProt accession IDs from ProtNote FASTA files to structure files with steps:
    1. Check for PDB cross-reference and download structure from RCSB PDB;
    2. If no PDB cross-reference or PDB does not cover whole sequence, check for AlphaFoldDB cross-reference;
    3. Save structure along with UniProt accession ID to json file.

Usage:
    python download_structure.py <uniprot_pkl_file>
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from protnote.utils.configs import load_config
from protnote.utils.network import fetch_with_retries

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG, ROOT_PATH = load_config()
DATA_PATH = ROOT_PATH / "data"


def parse_alphafold_db_api(accession_id: str) -> tuple[str, ...] | None:
    """Parse AlphaFoldDB download URL from API.

    Args:
        accession_id (str): The accession ID of the structure.

    Returns:
        tuple[str, ...] | None: A tuple containing the model entity ID, avg pLDDT, and CIF URL, or None if an error occurred.
    """
    url = f"{CONFIG['remote_data']['ALPHAFOLD_DB_API']}/{accession_id}"
    try:
        response = fetch_with_retries(url)
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error fetching AlphaFoldDB data for {accession_id}: {e}")
        return None

    data = response.json()
    if isinstance(data, list):
        data = data[0]
    return (data["modelEntityId"], data["globalMetricValue"], data["cifUrl"])


def download_pdb_structure(seq_id: str, pdb_id: str, output_dir: Path, override: bool = False) -> Path | None:
    """Download a structure file from RCSB PDB in CIF format.
    If the structure is not available in CIF format, try downloading it in PDB format.

    Args:
        seq_id (str): The sequence ID of the structure.
        pdb_id (str): The PDB ID of the structure.
        output_dir (Path): The directory to save the structure file.
        override (bool): Whether to override the existing file.
    """
    filename = f"{seq_id.lower()}.cif"
    output_path = output_dir / filename

    if output_path.exists() and not override:
        return output_path

    url = f"{CONFIG['remote_data']['RCSB_PDB_URL']}/{pdb_id.upper()}.cif"
    fallback_url = f"{CONFIG['remote_data']['RCSB_PDB_URL']}/{pdb_id.upper()}.pdb"

    try:
        response = fetch_with_retries(url)
    except requests.exceptions.RequestException as e:
        logger.warning(f"HTTP error fetching RCSB PDB data for {seq_id} with {pdb_id}: {e}")
        try:
            response = fetch_with_retries(fallback_url)
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching RCSB PDB data for {seq_id} with {pdb_id} (fallback): {e}")
            return None

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def download_alphafold_db_structure(seq_id: str, url: str, output_dir: Path, override: bool = False) -> Path | None:
    """Download a structure file from AlphaFoldDB in CIF format.

    Args:
        seq_id (str): The sequence ID of the structure.
        url (str): The AlphaFoldDB URL of the structure file.
        output_dir (Path): The directory to save the structure file.
        override (bool): Whether to override the existing file.
    """
    filename = f"{seq_id.lower()}.cif"
    output_path = output_dir / filename

    if output_path.exists() and not override:
        return output_path

    try:
        response = fetch_with_retries(url)
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error fetching AlphaFoldDB data for {seq_id}: {e}")
        return None

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def _append_alphafold_db_infos_to_df(df: pd.DataFrame) -> None:
    """Append AlphaFoldDB information to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to append information to.
    """
    df["afdb_model"] = None
    df["afdb_avg_plddt"] = None
    df["afdb_url"] = None

    # Mask for entries with AlphaFoldDB IDs
    mask = df["struct_afdb"].notna()

    for idx in df[mask].index:
        afdb_id = df.loc[idx, "struct_afdb"]
        afdb_info = parse_alphafold_db_api(afdb_id)
        if not afdb_info:
            logging.warning(f"Failed to fetch AlphaFoldDB data for {afdb_id}. Skipping.")
            continue
        df.loc[idx, "afdb_model"] = afdb_info[0]
        df.loc[idx, "afdb_avg_plddt"] = afdb_info[1]
        df.loc[idx, "afdb_url"] = afdb_info[2]


def main():
    parser = argparse.ArgumentParser(description="Download structure files given a UniProt pickle file.")
    parser.add_argument("uniprot_pkl_file", type=Path, help="Path to the UniProt pickle file.")
    parser.add_argument("--alphafolddb", action="store_true", default=False, help="Download only AlphaFoldDB structures.")
    parser.add_argument("--override", action="store_true", help="Override existing files.")

    args = parser.parse_args()

    # Setup output directories
    structure_dir = CONFIG["paths"]["data_paths"].get("STRUCTURE_DIR", DATA_PATH / "structures")
    pdb_dir = Path(structure_dir) / "pdb"
    af_dir = Path(structure_dir) / "alphafolddb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    af_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index if available
    index_path = CONFIG["paths"]["data_paths"].get("STRUCTURE_INDEX_PATH", DATA_PATH / "structures" / "structure_index.json")
    if index_path.exists():
        with open(index_path) as f:
            structure_index = json.load(f)
        logger.info(f"Loaded existing index with {len(structure_index)} entries")
    else:
        structure_index = {}

    # Load the UniProt pickle file
    uniprot_df = pd.read_pickle(args.uniprot_pkl_file)
    sequence_ids = set(uniprot_df["seq_id"])

    # Parse AlphaFoldDB information
    if "afdb_url" not in uniprot_df.columns or args.override:
        _append_alphafold_db_infos_to_df(uniprot_df)

    # Filter to only unprocessed IDs
    if not args.override:
        remaining = sorted(sequence_ids - set(structure_index.keys()))
    else:
        remaining = sorted(sequence_ids)
    logger.info(f"{len(remaining)} sequence IDs to process ({len(structure_index)} already indexed)")

    # Download structure files
    for row in tqdm(uniprot_df.itertuples(), total=len(uniprot_df)):
        output_path = None
        accession_id = row.seq_id

        # Skip if already processed and not overriding
        if row.seq_id not in remaining and not args.override:
            continue

        # Perform structure download
        if row.struct_expr and not args.alphafolddb:
            pdb = row.struct_expr[0]  # TODO: Handle multiple structures
            try:
                output_path = download_pdb_structure(
                    seq_id=accession_id,
                    pdb_id=pdb[0],
                    output_dir=pdb_dir,
                    override=args.override,
                )
                logging.info(f"Downloaded PDB structure {pdb[0]} for {accession_id}")
                if output_path is not None:
                    chain_ids = pdb[1].split("/")
                    structure_index[accession_id] = {
                        "source": "pdb",
                        "path": f"pdb/{output_path.name}",
                        "pdb_id": pdb[0],
                        "chain_ids": chain_ids,
                    }
            except Exception as e:
                logging.warning(f"PDB {pdb[0]} download failed for {accession_id}: {e}")

        if output_path is None and row.struct_afdb:
            try:
                output_path = download_alphafold_db_structure(
                    seq_id=accession_id,
                    url=row.afdb_url,
                    output_dir=af_dir,
                    override=args.override,
                )
                logging.info(f"Downloaded AlphaFoldDB structure {row.afdb_model} for {accession_id}")
                if output_path is not None:
                    structure_index[accession_id] = {
                        "source": "alphafolddb",
                        "path": f"alphafolddb/{output_path.name}",
                    }
            except Exception as e:
                logging.warning(f"AlphaFoldDB {row.afdb_model} download failed for {accession_id}: {e}")
            if output_path is None:
                logging.warning(f"No structure found for {accession_id}")

        # Periodically save the index to disk
        if (row.Index + 1) % 500 == 0:
            with open(index_path, "w") as f:
                json.dump(structure_index, f, indent=2)
            logger.info(f"Saved intermediate index ({len(structure_index)} entries)")

    # Finalize the index
    with open(index_path, "w") as f:
        json.dump(structure_index, f, indent=2)
    logger.info(f"Saved final index ({len(structure_index)} entries)")

    # Save the updated DataFrame
    output_path = args.uniprot_pkl_file.parent / "uniprot_structures.pkl"
    uniprot_df.to_pickle(output_path)

    logging.info(f"Saved updated DataFrames to {output_path}")


if __name__ == "__main__":
    main()
