"""Helper functions for data directory syncing."""

import argparse
import logging
import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, login, snapshot_download
from tqdm import tqdm

from protnote.utils.configs import load_config

config, project_root = load_config()
datapath_remote = config["paths"]["remote_data"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fetch_data_published() -> None:
    """Fetch ProtNote's published data & model from Zenodo."""
    # Download original data from Zenodo
    response = requests.get(datapath_remote["ZENODO_ORIGINAL_DATA"], stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(project_root / "data.zip", "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()

    # Unzip the downloaded file to data directory
    with zipfile.ZipFile(project_root / "data.zip", "r") as zip_ref:
        zip_ref.extractall(project_root / "data")
    (project_root / "data.zip").unlink()


def data_upload(repo_id: str) -> None:
    """Upload data directory to HuggingFace Hub Repository."""
    api = HfApi()
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=project_root / "data")


def data_download(repo_id: str, original: bool = False) -> None:
    """Download data and save to the data directory."""
    if original:
        _fetch_data_published()
    else:
        # Download data from HuggingFace Hub Repository
        snapshot_download(repo_id, local_dir=project_root / "data")


if __name__ == "__main__":
    Path(project_root / "data").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="ProtNote data helper")
    parser.add_argument("action", choices=["upload", "download"], help="Action: upload / download")
    parser.add_argument("--original", action="store_true", help="Download original published data")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace repository ID")
    args = parser.parse_args()

    if not args.original:
        # HuggingFace login
        login(new_session=False)

        # Fetch data repo id from config file if not provided
        if args.repo_id is None:
            args.repo_id = datapath_remote["HUGGINGFACE_DATA_REPO"]

    if args.action == "upload":
        data_upload(args.repo_id)
    elif args.action == "download":
        data_download(args.repo_id, args.original)
