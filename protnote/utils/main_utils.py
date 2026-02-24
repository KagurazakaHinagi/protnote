import torch
from tqdm import tqdm
from protnote.data.collators import collate_variable_sequence_length
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd
from functools import partial
import warnings

from omegaconf import DictConfig


def validate_arguments(cfg: DictConfig):
    run = cfg.run

    # Ensure the full data path is provided, or we are using the zero shot model
    if run.full_path_name is None and "zero" not in str(run.train_path_name).lower():
        warnings.warn(
            "The full path name is not provided and the train path name does not contain the word 'zero'. Please ensure this is intentional."
        )

    # Raise error if train is provided without val
    if run.train_path_name is not None:
        if run.validation_path_name is None:
            raise ValueError(
                "If providing train_path_name you must provide validation_path_name."
            )

    # Raise error if no train path is provided and no model is loaded
    if (run.train_path_name is None) and (run.model_file is None):
        raise ValueError(
            "You must provide model_file if no train_path_name is provided"
        )

    # Raise error if none of the paths are provided

    if (
        (run.test_paths_names is None)
        and (run.train_path_name is None)
        and (run.validation_path_name is None)
    ):
        raise ValueError(
            "You must provide one of the following options:\n"
            "run.test_paths_names + run.model_file\n"
            "run.validation_path_name + run.model_file\n"
            "run.train_path_name and run.validation_path_name (optional model_file)\n"
            "run.train_path_name and run.validation_path_name + run.test_paths_names (optional model_file)\n"
            "All cases with including run.full_path_name. Please provide the required option(s) and try again."
        )

    if (run.save_prediction_results) and (
        (run.test_paths_names is None) and (run.validation_path_name is None)
    ):
        raise ValueError(
            "You must provide test_paths_names and/or validation_path_name to save the results of the validation and/or test sets."
        )


def generate_sequence_embeddings(device, sequence_encoder, datasets, params):
    """Generate sequence embeddings for the given datasets."""
    sequence_encoder = sequence_encoder.to(device)
    sequence_encoder.eval()
    all_datasets = [
        dataset for dataset_list in datasets.values() for dataset in dataset_list
    ]
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
        shuffle=False,
        collate_fn=partial(
            collate_variable_sequence_length, return_label_multihots=False
        ),  # have to use return_label_multihots to ignore multihot concat with zero shot
        num_workers=params["NUM_WORKERS"],
        pin_memory=True,
    )
    # Initialize an empty list to store data
    data_list = []

    for batch in tqdm(combined_loader):
        sequence_onehots, sequence_ids, sequence_lengths = (
            batch["sequence_onehots"].to(device),
            batch["sequence_ids"],
            batch["sequence_lengths"].to(device),
        )
        with torch.no_grad():
            embeddings = sequence_encoder.get_embeddings(
                sequence_onehots, sequence_lengths
            )
            for i, original_id in enumerate(sequence_ids):
                data_list.append((original_id, embeddings[i].cpu().numpy()))

    sequence_encoder.train()
    # Convert the list to a DataFrame
    df = pd.DataFrame(data_list, columns=["ID", "Embedding"]).set_index("ID")
    return df
