"""Protein structure processing utilities.

Functions for parsing structure files, building atom-level representations,
and aligning structures with sequence embeddings."""

import logging
from os import PathLike
from pathlib import Path

import atomworks
import atomworks.constants as awconst
import atomworks.io as awio
import biotite.structure as bs
import numpy as np
import numpy.typing as npt
import torch
from atomworks.io.parser import parse_atom_array
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.utils.token import get_token_starts
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def element_to_index(element: str) -> int:
    """Converts an element symbol to its atomic number index."""
    return awconst.ELEMENT_NAME_TO_ATOMIC_NUMBER.get(element, 0)


def residue_3_to_1(res_name3: str) -> str:
    """Converts a residue name from 3-letter to 1-letter code."""
    return awconst.DICT_THREE_TO_ONE.get(res_name3, "X")


def _impute_nan_coords(atom_array) -> tuple:
    """Impute NaN coordinates using nearby resolved atoms.

    For atoms whose coordinates are NaN (added by AtomWorks from CCD templates
    but spatially unresolved):
      - Partially resolved residues: impute at the residue's own resolved centroid.
      - Fully unresolved residues: impute at the centroid of the nearest resolved
        neighbor residue (by res_id within the same chain), preserving sequence
        contiguity for downstream ESM-C alignment.

    Args:
        atom_array: Biotite AtomArray, potentially with NaN coordinates.

    Returns:
        Tuple of (cleaned AtomArray, stats dict with imputation counts).
    """
    coords = atom_array.coord
    nan_mask = np.isnan(coords).any(axis=1)
    n_nan = int(nan_mask.sum())
    n_total = atom_array.array_length()

    if n_nan == 0:
        return atom_array, {"n_total": n_total, "n_imputed": 0, "n_removed": 0, "n_neighbor_imputed": 0}

    # Group atoms by (chain_id, res_id)
    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    res_keys = np.array([f"{c}_{r}" for c, r in zip(chain_ids, res_ids)])
    unique_res = np.unique(res_keys)

    # First pass: compute centroid for each residue that has resolved atoms
    res_centroids = {}  # res_key -> centroid coords
    fully_unresolved_keys = []
    for res_key in unique_res:
        res_mask = res_keys == res_key
        res_resolved = res_mask & ~nan_mask
        if res_resolved.any():
            res_centroids[res_key] = coords[res_resolved].mean(axis=0)
        else:
            fully_unresolved_keys.append(res_key)

    imputed = 0
    neighbor_imputed = 0

    # Second pass: impute partially resolved residues from own centroid
    for res_key in unique_res:
        res_mask = res_keys == res_key
        res_nan = nan_mask & res_mask
        if not res_nan.any():
            continue
        if res_key in res_centroids:
            coords[res_nan] = res_centroids[res_key]
            imputed += int(res_nan.sum())

    # Third pass: impute fully unresolved residues from nearest resolved neighbor
    for res_key in fully_unresolved_keys:
        res_mask = res_keys == res_key
        res_nan = nan_mask & res_mask
        # Parse chain and res_id
        chain, rid_str = res_key.rsplit("_", 1)
        rid = int(rid_str)

        # Find nearest resolved residue in the same chain by res_id distance
        best_dist = float("inf")
        best_centroid = None
        for other_key, centroid in res_centroids.items():
            other_chain, other_rid_str = other_key.rsplit("_", 1)
            if other_chain != chain:
                continue
            dist = abs(int(other_rid_str) - rid)
            if dist < best_dist:
                best_dist = dist
                best_centroid = centroid

        if best_centroid is not None:
            coords[res_nan] = best_centroid
            neighbor_imputed += int(res_nan.sum())
        else:
            # No resolved residues in the entire chain; leave as NaN
            # (will be caught by downstream validation)
            logger.warning(f"No resolved residues in chain {chain} to impute from")

    atom_array.coord = coords

    # Remove any remaining NaN atoms (only if an entire chain had no resolved atoms)
    still_nan = np.isnan(atom_array.coord).any(axis=1)
    n_removed = int(still_nan.sum())
    if n_removed > 0:
        atom_array = atom_array[~still_nan]

    stats = {
        "n_total": n_total,
        "n_imputed": imputed,
        "n_neighbor_imputed": neighbor_imputed,
        "n_removed": n_removed,
    }
    return atom_array, stats


def parse_structure(cif_path: PathLike):
    """Parse a structure file into a cleaned AtomArray.

    Uses AtomWorks.io to parse CIF/PDB files, removing hydrogens, waters,
    and atoms with NaN coordinates.

    Args:
        cif_path (PathLike): Path to the CIF/PDB file.

    Returns:
        Biotite AtomArray with cleaned atom data and bonds.
    """

    raw_atom_array = load_any(cif_path, extra_fields=["auth_asym_id"])

    # Replace label chain IDs with author chain IDs if available
    # to match the chain ID info in UniProt.
    if hasattr(raw_atom_array, "auth_asym_id"):
        raw_atom_array.chain_id = raw_atom_array.auth_asym_id

    parsed_result = parse_atom_array(
        raw_atom_array,
        hydrogen_policy="remove",
    )

    atom_array = parsed_result["asym_unit"][0]

    # Impute NaN coordinates instead of discarding them to preserve sequence contiguity
    atom_array, stats = _impute_nan_coords(atom_array)
    if stats["n_imputed"] > 0 or stats["n_neighbor_imputed"] > 0 or stats["n_removed"] > 0:
        logger.info(
            f"{Path(cif_path).stem}: imputed {stats['n_imputed']} atoms at residue centroids, "
            f"{stats['n_neighbor_imputed']} atoms at neighbor residue centroids, "
            f"removed {stats['n_removed']} (total atoms: {stats['n_total']})"
        )

    return atom_array


def extract_aa_residue_by_chain_ids(atom_array: bs.AtomArray, chain_ids: npt.ArrayLike | None = None) -> bs.AtomArray:
    """Extract a subset of the amino acid atom array by chain IDs.

    Args:
        atom_array (bs.AtomArray): Atom array to extract from.
        chain_ids (ArrayLike | None): List of chain IDs to extract.

    Returns:
        AtomArray with only the amino acid atoms from the specified chain IDs.
    """
    residue_mask = bs.filter_amino_acids(atom_array)
    if chain_ids:
        chain_mask = atom_array.chain_id == chain_ids[0]
        mask = chain_mask & residue_mask
    else:
        mask = residue_mask
    return atom_array[mask]


def _get_residue_mapping(atom_array: bs.AtomArray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build residue index array and residue name list from atom array.

    Returns:
        residue_index: LongTensor-compatible array mapping each atom to its residue index.
        residue_names: List of residue names.
        residue_res_ids: Array of res_id values
    """

    token_starts = get_token_starts(atom_array, add_exclusive_stop=True)
    n_residues = len(token_starts) - 1
    n_atoms = atom_array.array_length()

    residue_index = np.zeros(n_atoms, dtype=np.int64)
    residue_names = []
    residue_res_ids = np.zeros(n_atoms, dtype=np.int64)

    for i in range(n_residues):
        start = token_starts[i]
        end = token_starts[i + 1]
        residue_index[start:end] = i
        residue_names.append(atom_array.res_name[start])
        residue_res_ids[i] = atom_array.res_id[start]

    return residue_index, residue_names, residue_res_ids


def _extract_covalent_edges(atom_array: bs.AtomArray) -> np.ndarray:
    """Extract covalent bond edges from atom array.

    Returns:
        edge_index: [2, E_cov] array of covalent edges.
    """
    if atom_array.bonds is None:
        return np.zeros((2, 0), dtype=np.int64)

    bond_array = atom_array.bonds.as_array()
    if len(bond_array) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # bond_array columns: [atom_i, atom_j, bond_type]
    src = bond_array[:, 0]
    dst = bond_array[:, 1]
    # Make bidirectional, ignore bond type
    edge_index = np.stack(
        [
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ],
        axis=0,
    ).astype(np.int64)

    return edge_index


def _build_knn_edges(coords: np.ndarray, k: int = 20) -> np.ndarray:
    """Build k-NN spatial edges from atom coordinates.

    Args:
        coords: [N, 3] array of atom coordinates.
        k: Number of nearest neighbors to consider.

    Returns:
        edge_index: [2, E_knn] array of k-NN edges.
    """
    n_atoms = coords.shape[0]
    if n_atoms <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    actual_k = min(k + 1, n_atoms)
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=actual_k)

    # Build edge list
    src_list = []
    dst_list = []
    for i in range(n_atoms):
        for j_idx in range(actual_k):
            j = indices[i, j_idx]
            if i != j:
                src_list.append(i)
                dst_list.append(j)

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=0)

    return edge_index


def build_protein_atom_graph(atom_array: bs.AtomArray, chains: npt.ArrayLike | None = None, k: int = 20) -> dict[str, any]:
    """Build an atom-level graph for protein segments from a Biotite AtomArray.

    Constructs coordinates, atom features, residue mapping,
    and combined covalent + k-NN edges.

    Args:
        atom_array: Biotite AtomArray (cleaned, filtered, w/o hydrogens/waters)
        chains: List of chain IDs to include in the graph. If None, all chains are included.
        k: Number of nearest neighbors to consider for k-NN edges

    Returns:
        Dictionary with graph data:
            - coords: [N_atoms, 3] float32
            - atom_types: [N_atoms] int64 (element-based indices)
            - atom_names: [N_atoms] str (name3)
            - residue_index: [N_atoms] int64
            - residue_names: [N_atoms] str
            - residue_res_ids: [N_residues] int64
            - edge_index: [2, E] int64
            - edge_type: [E] int64 (0=covalent, 1=k-NN)
            - n_atoms: int
            - n_residues: int
    """

    coords = atom_array.coord.astype(np.float32)
    n_atoms = coords.shape[0]

    # Atom types from element symbols
    atom_types = np.array([element_to_index(e) for e in atom_array.element], dtype=np.int64)

    # Atom names
    atom_names = list(atom_array.atom_name)

    # Residue mapping
    residue_index, residue_names, residue_res_ids = _get_residue_mapping(atom_array)
    n_residues = len(residue_names)

    # Build edges
    cov_edges = _extract_covalent_edges(atom_array)
    knn_edges = _build_knn_edges(coords, k)

    # Combine and deduplicate
    if cov_edges.shape[1] > 0 and knn_edges.shape[1] > 0:
        all_edges = np.concatenate([cov_edges, knn_edges], axis=1)
        edge_types = np.concatenate([np.zeros(cov_edges.shape[1], dtype=np.int64), np.ones(knn_edges.shape[1], dtype=np.int64)])

        # Deduplicate: for edges that appear in both covalent and k-NN
        # keep the covalent lavel (type=0)
        edge_set = {}
        for idx in range(all_edges.shape[1]):
            key = (all_edges[0, idx], all_edges[1, idx])
            if key not in edge_set or edge_types[idx] == 0:
                edge_set[key] = idx

        if edge_set:
            dedup_edges = np.array(list(edge_set.keys()), dtype=np.int64).T
            dedup_types = np.array(list(edge_set.values()), dtype=np.int64)
        else:
            dedup_edges = np.zeros((2, 0), dtype=np.int64)
            dedup_types = np.zeros(0, dtype=np.int64)

    elif cov_edges.shape[1] > 0:
        dedup_edges = cov_edges
        dedup_types = np.zeros(cov_edges.shape[1], dtype=np.int64)

    elif knn_edges.shape[1] > 0:
        dedup_edges = knn_edges
        dedup_types = np.ones(knn_edges.shape[1], dtype=np.int64)

    else:
        dedup_edges = np.zeros((2, 0), dtype=np.int64)
        dedup_types = np.zeros(0, dtype=np.int64)

    return {
        "coords": coords,
        "atom_types": atom_types,
        "atom_names": atom_names,
        "residue_index": residue_index,
        "residue_names": residue_names,
        "residue_res_ids": residue_res_ids,
        "edge_index": dedup_edges,
        "edge_type": dedup_types,
        "n_atoms": n_atoms,
        "n_residues": n_residues,
    }


def get_sequence_from_residues(residue_names: list[str]) -> str:
    """Convert a list of 3-letter residue names to a 1-letter sequence string."""
    return "".join([residue_3_to_1(rn) for rn in residue_names])


def _get_residue_list(atom_array) -> list[tuple[int, str]]:
    """Get ordered list of (res_id, res_name) for each residue in the atom array."""
    residues = []
    seen = set()
    for rid, rname, cid in zip(atom_array.res_id, atom_array.res_name, atom_array.chain_id):
        key = (int(rid), str(rname), str(cid))
        if key not in seen:
            seen.add(key)
            residues.append(key)
    return residues


def trim_terminal_tags(
    atom_array,
    fasta_sequence: str,
    chain_id: str | None = None,
    max_mismatch_rate: float = 0.02,
) -> tuple:
    """Trim expression tags and cloning artifacts from structure terminals.

    Compares the structure-derived amino acid sequence against the FASTA
    reference to detect non-matching terminal residues (His-tags, expression
    tags, TEV cleavage remnants, etc.) and removes them from the AtomArray.

    Only trims the specified protein chain; non-protein chains (ligands,
    waters, ions) are left untouched.

    The algorithm:
      1. If structure sequence length <= FASTA length, no tags to trim.
      2. Try exact substring: find FASTA as contiguous substring of structure.
      3. If exact fails (due to rare non-standard residues), try sliding-window
         alignment allowing a small mismatch rate.

    Args:
        atom_array: Biotite AtomArray (full structure, may contain multiple chains).
        fasta_sequence: Reference FASTA sequence for the protein.
        chain_id: Which chain to trim. If None, auto-detects the longest
            protein chain.
        max_mismatch_rate: Maximum fraction of mismatches to tolerate in the
            fuzzy alignment (default 2%, handles rare non-standard residues).

    Returns:
        Tuple of (trimmed AtomArray, n_trimmed_nterm, n_trimmed_cterm).
        If no trimming is needed or alignment fails, returns the original
        AtomArray with (0, 0).
    """
    # Identify the target chain
    if chain_id is None:
        # Pick the chain with the most atoms (likely the protein)
        unique_chains = np.unique(atom_array.chain_id)
        chain_id = max(unique_chains, key=lambda c: (atom_array.chain_id == c).sum())

    chain_mask = atom_array.chain_id == chain_id
    chain_atoms = atom_array[chain_mask]

    if chain_atoms.array_length() == 0:
        return atom_array, 0, 0

    # Build per-residue sequence from the target chain
    residues = [(rid, rname) for rid, rname, cid in _get_residue_list(atom_array) if cid == chain_id]
    struct_seq = "".join(residue_3_to_1(rname) for _, rname in residues)

    n_struct = len(struct_seq)
    n_fasta = len(fasta_sequence)

    if n_struct <= n_fasta:
        # Structure is same length or shorter — no terminal tags
        return atom_array, 0, 0

    n_extra = n_struct - n_fasta

    # Strategy 1: Exact substring match (FASTA found within structure)
    idx = struct_seq.find(fasta_sequence)
    if idx >= 0:
        n_trim_n = idx
        n_trim_c = n_struct - idx - n_fasta
        return _apply_residue_trim(atom_array, chain_id, residues, n_trim_n, n_trim_c)

    # Strategy 2: Sliding-window fuzzy alignment
    max_mismatches = max(1, int(n_fasta * max_mismatch_rate))
    best_offset = -1
    best_mismatches = n_fasta + 1

    for offset in range(n_extra + 1):
        mismatches = 0
        for i in range(n_fasta):
            s_aa = struct_seq[offset + i]
            f_aa = fasta_sequence[i]
            if s_aa != f_aa and s_aa != "X" and f_aa != "X":
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_offset = offset

    if best_mismatches <= max_mismatches:
        n_trim_n = best_offset
        n_trim_c = n_struct - best_offset - n_fasta
        return _apply_residue_trim(atom_array, chain_id, residues, n_trim_n, n_trim_c)

    logger.warning(
        f"Cannot align structure ({n_struct} res) to FASTA ({n_fasta} res) "
        f"for tag trimming (best mismatch: {best_mismatches}). Skipping trim."
    )
    return atom_array, 0, 0


def _apply_residue_trim(
    atom_array,
    chain_id: str,
    residues: list[tuple[int, str]],
    n_trim_n: int,
    n_trim_c: int,
):
    """Remove N-terminal and C-terminal residues from a specific chain.

    Args:
        atom_array: Full AtomArray.
        chain_id: Chain to trim.
        residues: Ordered list of (res_id, res_name) for the chain.
        n_trim_n: Number of residues to remove from N-terminus.
        n_trim_c: Number of residues to remove from C-terminus.

    Returns:
        Tuple of (trimmed AtomArray, n_trim_n, n_trim_c).
    """
    if n_trim_n == 0 and n_trim_c == 0:
        return atom_array, 0, 0

    # Collect res_ids to remove
    remove_res_ids = set()
    if n_trim_n > 0:
        for rid, _ in residues[:n_trim_n]:
            remove_res_ids.add(rid)
    if n_trim_c > 0:
        for rid, _ in residues[len(residues) - n_trim_c :]:
            remove_res_ids.add(rid)

    # Build atom-level mask: keep atoms NOT in the removed residues of this chain
    keep_mask = np.ones(atom_array.array_length(), dtype=bool)
    for i in range(atom_array.array_length()):
        if atom_array.chain_id[i] == chain_id and int(atom_array.res_id[i]) in remove_res_ids:
            keep_mask[i] = False

    trimmed_names = []
    if n_trim_n > 0:
        trimmed_names.append(f"N-term: {[r[1] for r in residues[:n_trim_n]]}")
    if n_trim_c > 0:
        trimmed_names.append(f"C-term: {[r[1] for r in residues[len(residues) - n_trim_c :]]}")
    logger.info(f"Trimmed {n_trim_n + n_trim_c} tag residues from chain {chain_id} ({', '.join(trimmed_names)})")

    return atom_array[keep_mask], n_trim_n, n_trim_c


def align_esmc_to_structure(
    esmc_emb: torch.Tensor,
    structure_res_names: list[str],
    fasta_seq: str | None = None,
    subalign: bool = True,
) -> torch.Tensor | None:
    """Align ESM-C per-residue embeddings to structure residues.

    ESM-C embeddings correspond to the FASTA sequence, while the structure
    may have missing residues or unresolved residues. This function supports:
        - Exact length match (no alignment needed)
        - Contiguous substring match (terminal truncations only)

    Args:
        esmc_emb: [L_esmc, 960] ESM-C embeddings.
        structure_res_names: List of residue names from the structure.
        fasta_seq: Original FASTA sequence (optional, for logging).
        subalign: Whether to perform subalignment if lengths mismatch (default: True).
    Returns:
        Aligned embeddings [N_res_structure, 960], or None if alignment fails.
    """
    n_struct_res = len(structure_res_names)
    n_esmc_res = esmc_emb.shape[0]

    if n_struct_res == n_esmc_res:
        return esmc_emb

    # If the structure has fewer residues than ESM-C (missing residues in structure),
    # we need to find the matching positions
    if fasta_seq is not None and n_esmc_res == len(fasta_seq):
        struct_seq = get_sequence_from_residues(structure_res_names)

        # Try to find structure sequence as a subsequence of FASTA sequence
        # This handles cases where structure is missing terminal residues
        idx = fasta_seq.find(struct_seq)
        if idx >= 0:
            logger.info("Found structure sequence as a subsequence of FASTA sequence.")
            logger.info(f"Alignment residue indices: {idx} to {idx + n_struct_res}")
            if subalign:
                logger.info("Performing subalignment.")
                return esmc_emb[idx : idx + n_struct_res]
            else:
                logger.warning("Subalignment disabled. Skipping alignment.")
                return None

    logger.warning(f"Structure has {n_struct_res} residues, ESM-C has {n_esmc_res}.")
    logger.warning("Alignment failed.")

    return None
