from typing import List, Tuple

import torch

try:
    from torch_geometric.data import Batch as PyGBatch
except ImportError:
    PyGBatch = None


def collate_structure_batch(
    batch: List[dict],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    world_size=1,
    rank=0,
    return_label_multihots=True,
):
    """
    Collate for structural encoder. Supports two formats:

    Legacy format (PyG):
        Input items have 'structure_batch' (PyG Data)
        Returns dict with 'structure_batch' (PyG Batch)

    Atom-level format:
        Input items have 'atom_coords', 'atom_types', 'esmc_embeddings', etc.
        Returns dict with batched tensors + 'graph_data' dict for encoder
    """
    # Detect format based on first item
    if "structure_batch" in batch[0]:
        return _collate_legacy(
            batch, label_sample_size, distribute_labels, shuffle_labels, in_batch_sampling, grid_sampler, world_size, rank
        )
    else:
        return _collate_atom_level(
            batch,
            label_sample_size,
            distribute_labels,
            shuffle_labels,
            in_batch_sampling,
            grid_sampler,
            world_size,
            rank,
            return_label_multihots,
        )


def _collate_legacy(
    batch: List[dict],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    world_size=1,
    rank=0,
):
    """
    Collate for legacy PyG format: list of dicts with structure_batch (PyG Data).
    Returns dict with structure_batch (PyG Batch), sequence_ids, label_multihots, etc.
    """
    if PyGBatch is None:
        raise ImportError("collate_structure_batch requires torch_geometric: pip install torch-geometric")
    graphs = [item["structure_batch"] for item in batch]
    structure_batch = PyGBatch.from_data_list(graphs)
    sequence_ids = [item["sequence_id"] for item in batch]
    label_multihots = torch.stack([item["label_multihots"] for item in batch])
    label_embeddings = batch[0]["label_embeddings"]
    label_token_counts = batch[0]["label_token_counts"]

    if label_sample_size and not grid_sampler and not in_batch_sampling:
        num_labels = label_multihots.shape[1]
        if distribute_labels:
            labels_per_partition = num_labels // world_size
            start_idx = rank * labels_per_partition
            end_idx = start_idx + labels_per_partition
            sampled_label_indices = torch.arange(start_idx, end_idx)[: label_sample_size // world_size]
        else:
            sampled_label_indices = (
                torch.randperm(num_labels)[:label_sample_size] if shuffle_labels else torch.arange(min(label_sample_size, num_labels))
            )
        label_embeddings = label_embeddings[sampled_label_indices]
        label_multihots = label_multihots[:, sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]
    elif in_batch_sampling:
        sampled_label_indices = torch.where(label_multihots.sum(dim=0) > 0)[0]
        label_embeddings = label_embeddings[sampled_label_indices]
        label_multihots = label_multihots[:, sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]

    return {
        "structure_batch": structure_batch,
        "sequence_ids": sequence_ids,
        "label_multihots": label_multihots,
        "label_embeddings": label_embeddings,
        "label_token_counts": label_token_counts,
    }


def _collate_atom_level(
    batch: List[dict],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    world_size=1,
    rank=0,
    return_label_multihots=True,
):
    """
    Collate for atom-level format: batches atom-level graphs.

    Returns dict with:
        - graph_data: dict for encoder (esmc_embeddings, atom_coords, atom_types,
                      edge_index, atom_to_protein, num_proteins)
        - sequence_ids, label_multihots, label_embeddings, label_token_counts
    """
    # Batch graph data (PyG-style: concatenate with offsets)
    all_atom_coords = []
    all_atom_types = []
    all_esmc_embeddings = []
    all_residue_indices = []
    all_edge_index = []
    all_atom_to_protein = []
    sequence_ids = []

    atom_offset = 0

    for i, item in enumerate(batch):
        n_atoms = item["num_atoms"]

        all_atom_coords.append(item["atom_coords"])
        all_atom_types.append(item["atom_types"])

        # Broadcast esmc_embeddings from per-residue to per-atom if needed
        esmc_emb = item["esmc_embeddings"]
        atom_to_res = item["atom_to_residue"]
        if esmc_emb.size(0) != n_atoms:
            # Per-residue format: broadcast to atoms
            esmc_emb_per_atom = esmc_emb[atom_to_res]
        else:
            esmc_emb_per_atom = esmc_emb
        all_esmc_embeddings.append(esmc_emb_per_atom)

        if "residue_indices" in item:
            all_residue_indices.append(item["residue_indices"])

        # Offset edge indices by cumulative atom count
        edge_index = item["edge_index"]
        if edge_index.numel() > 0:
            all_edge_index.append(edge_index + atom_offset)
        else:
            all_edge_index.append(edge_index)

        # Track which atoms belong to which protein
        all_atom_to_protein.append(torch.full((n_atoms,), i, dtype=torch.long))

        sequence_ids.append(item["sequence_id"])
        atom_offset += n_atoms

    # Concatenate all tensors
    batched_atom_coords = torch.cat(all_atom_coords, dim=0)
    batched_atom_types = torch.cat(all_atom_types, dim=0)
    batched_esmc_embeddings = torch.cat(all_esmc_embeddings, dim=0)
    batched_atom_to_protein = torch.cat(all_atom_to_protein, dim=0)

    if any(e.numel() > 0 for e in all_edge_index):
        non_empty = [e for e in all_edge_index if e.numel() > 0]
        batched_edge_index = torch.cat(non_empty, dim=1) if non_empty else torch.zeros(2, 0, dtype=torch.long)
    else:
        batched_edge_index = torch.zeros(2, 0, dtype=torch.long)

    batched_residue_indices = None
    if all_residue_indices:
        batched_residue_indices = torch.cat(all_residue_indices, dim=0)

    # Build graph_data dict for encoder
    graph_data = {
        "esmc_embeddings": batched_esmc_embeddings,
        "atom_coords": batched_atom_coords,
        "atom_types": batched_atom_types,
        "edge_index": batched_edge_index,
        "atom_to_protein": batched_atom_to_protein,
        "num_proteins": len(batch),
    }
    if batched_residue_indices is not None:
        graph_data["residue_indices"] = batched_residue_indices

    # Handle labels
    label_multihots = torch.stack([item["label_multihots"] for item in batch])
    label_embeddings = batch[0]["label_embeddings"]
    label_token_counts = batch[0]["label_token_counts"]

    # Label sampling (same logic as legacy)
    sampled_label_indices = None
    if label_sample_size and not grid_sampler and not in_batch_sampling:
        num_labels = label_multihots.shape[1]
        if distribute_labels:
            labels_per_partition = num_labels // world_size
            start_idx = rank * labels_per_partition
            end_idx = start_idx + labels_per_partition
            sampled_label_indices = torch.arange(start_idx, end_idx)[: label_sample_size // world_size]
        else:
            sampled_label_indices = (
                torch.randperm(num_labels)[:label_sample_size] if shuffle_labels else torch.arange(min(label_sample_size, num_labels))
            )
    elif in_batch_sampling:
        sampled_label_indices = torch.where(label_multihots.sum(dim=0) > 0)[0]

    if sampled_label_indices is not None:
        label_embeddings = label_embeddings[sampled_label_indices]
        label_multihots = label_multihots[:, sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]

    result = {
        "graph_data": graph_data,
        "sequence_ids": sequence_ids,
        "label_embeddings": label_embeddings,
        "label_token_counts": label_token_counts,
    }
    if return_label_multihots:
        result["label_multihots"] = label_multihots

    return result


def collate_variable_sequence_length(
    batch: List[Tuple],
    label_sample_size=None,
    distribute_labels=False,
    shuffle_labels=False,
    in_batch_sampling=False,
    grid_sampler=False,
    return_label_multihots=True,
    world_size=1,
    rank=0,
):
    """
    Collates a batch of data with variable sequence lengths. Pads sequences to the maximum length within the batch to handle the variable
    lengths.

    Args:
        batch (List[Tuple]): A list of tuples, where each tuple represents one data point.
                             Each tuple contains a dictionary with keys like 'sequence_onehots',
                             'sequence_length', 'label_multihots', etc.
        label_sample_size (int, optional): The number of labels to sample for training.
                                           Used with grid_sampler or in_batch_sampling.
        distribute_labels (bool, optional): Whether to distribute labels across different GPUs.
        return_label_multihots (bool, optional): Whether to batched multihot labels.
        shuffle_labels (bool, optional): Whether to shuffle labels during sampling.
        in_batch_sampling (bool, optional): If True, samples labels that are present within the batch.
        grid_sampler (bool, optional): If True, uses a grid sampling strategy for labels.
        world_size (int, optional): The total number of distributed processes or GPUs.
        rank (int, optional): The rank of the current process in distributed training.

    Returns:
        Dict: A dictionary containing the processed batch data. Keys include:
              - 'sequence_onehots': Tensor, padded one-hot encoded sequences.
              - 'sequence_ids': List, sequence IDs.
              - 'sequence_lengths': Tensor, lengths of sequences.
              - 'label_multihots': Tensor, multihot encoded labels (possibly sampled).
              - 'label_embeddings': Tensor, label embeddings if provided. Otherwise None.
              - 'label_token_counts': Tensor, token counts for each label.

    """

    # Determine the maximum sequence length in the batch
    max_length = max(item["sequence_length"] for item in batch)

    # Initialize lists to store the processed values
    processed_sequence_onehots = []
    processed_sequence_ids = []
    processed_sequence_lengths = []
    processed_label_multihots = []
    processed_label_token_counts = []
    processed_label_embeddings = None

    if grid_sampler:
        assert label_sample_size is not None, "Must provide label_sample_size if using grid sampler"
        assert not in_batch_sampling, "Can't use in batch sampling with grid sampler"

    else:
        assert not (in_batch_sampling and (label_sample_size is not None)), "Cant use both in_batch_sampling with lable_sample_size"

    sampled_label_indices = None
    num_labels = batch[0]["label_multihots"].shape[0]

    if label_sample_size:
        if grid_sampler:
            sampled_label_indices = batch[0]["label_idxs"]
        else:
            if not distribute_labels:
                # If not distributing labels, sample from entire dataset
                sampled_label_indices = (
                    torch.randperm(num_labels)[:label_sample_size] if shuffle_labels else torch.arange(min(label_sample_size, num_labels))
                )
            else:
                # Otherwise, sample from the labels on this GPU
                labels_per_partition = num_labels // world_size
                start_idx = rank * labels_per_partition
                end_idx = start_idx + labels_per_partition
                partition_indices = torch.arange(start_idx, end_idx)
                sampled_label_indices = partition_indices[torch.randperm(len(partition_indices))[: label_sample_size // world_size]]
                # if rank < 2:
                #     print("GPU {}. Sampling range: {} to {}. Sampled {} labels".format(rank, start_idx, end_idx, sampled_label_indices[:10]))

    elif in_batch_sampling:
        sampled_label_indices = torch.where(sum(i["label_multihots"] for i in batch) > 0)[0]

    # Apply the sampled labels to the label embeddings
    # We only use the first sequence in the batch to get the label embeddings to minimize complexity
    label_embeddings = batch[0]["label_embeddings"]

    # Likewise, we only use the first sequence in the batch to get the label token counts
    label_token_counts = batch[0]["label_token_counts"]

    if sampled_label_indices is not None:
        # Create a new tensor of embeddings with only the sampled labels
        processed_label_embeddings = label_embeddings[sampled_label_indices]
        label_token_counts = label_token_counts[sampled_label_indices]
    # Otherwise, use the original label embeddings
    else:
        processed_label_embeddings = label_embeddings

    # Loop through the batch
    for row in batch:
        # Get the sequence onehots, sequence length, sequence id, and label multihots
        sequence_onehots = row["sequence_onehots"]
        sequence_id = row["sequence_id"]
        sequence_length = row["sequence_length"]
        label_multihots = row["label_multihots"]

        # Set padding
        padding_length = max_length - sequence_length

        # Get the sequence dimension (e.g., 20 for amino acids)
        sequence_dim = sequence_onehots.shape[0]

        # Pad the sequence to the max_length and append to the processed_sequences list
        processed_sequence_onehots.append(torch.cat((sequence_onehots, torch.zeros((sequence_dim, padding_length))), dim=1))

        # Use the sampled labels for each element in the batch.
        if sampled_label_indices is not None:
            label_multihots = label_multihots[sampled_label_indices]

        # Append the other values to the processed lists
        processed_sequence_ids.append(sequence_id)
        processed_sequence_lengths.append(sequence_length)
        processed_label_multihots.append(label_multihots)

    processed_batch = {
        "sequence_onehots": torch.stack(processed_sequence_onehots),
        "sequence_ids": processed_sequence_ids,
        "sequence_lengths": torch.stack(processed_sequence_lengths),
        "label_embeddings": processed_label_embeddings,
        "label_token_counts": label_token_counts,
    }

    if return_label_multihots:
        processed_batch["label_multihots"] = torch.stack(processed_label_multihots)

    return processed_batch
