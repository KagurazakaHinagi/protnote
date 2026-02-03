"""
HybridProteinEncoder: Combines ESM-C sequence embeddings with EGNN structure
processing via hierarchical pooling (atom -> residue -> protein).

Replaces ProteInfer CNN as the protein encoder in the ToxinNote architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from protnote.models.protein_encoders import StructureEncoder


class HierarchicalPooling(nn.Module):
    """Two-stage pooling: atom -> residue (mean) -> protein (attention).

    Stage 1: Mean-pool atom features within each residue.
    Stage 2: Attention-weighted pool residue features into a single protein vector.

    The Stage 2 attention weights provide built-in residue-level explainability.

    Args:
        hidden_dim: Dimension of input atom/residue features.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Learnable attention scorer for residue-to-protein pooling
        self.attn_scorer = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, h_atoms, atom_to_residue, num_residues, residue_mask=None):
        """Pool atom features to a single protein vector.

        Args:
            h_atoms: Atom-level features [N_atoms, hidden_dim]
            atom_to_residue: Residue index for each atom [N_atoms]
            num_residues: Total number of residues (int)
            residue_mask: Optional boolean mask [num_residues] indicating valid residues

        Returns:
            h_protein: Protein-level feature vector [hidden_dim]
            attn_weights: Residue-level attention weights [num_residues]
        """
        hidden_dim = h_atoms.size(1)

        # Stage 1: Mean-pool atoms within each residue
        # Use scatter to group atoms by residue
        residue_ids = atom_to_residue.unsqueeze(-1).expand(-1, hidden_dim)
        h_residues = h_atoms.new_zeros(num_residues, hidden_dim)
        counts = h_atoms.new_zeros(num_residues, hidden_dim)
        h_residues.scatter_add_(0, residue_ids, h_atoms)
        counts.scatter_add_(0, residue_ids, torch.ones_like(h_atoms))
        h_residues = h_residues / counts.clamp(min=1)

        # Stage 2: Attention-weighted pooling over residues
        raw_scores = self.attn_scorer(h_residues).squeeze(-1)  # [num_residues]

        # Mask invalid residues (e.g., padding)
        if residue_mask is not None:
            raw_scores = raw_scores.masked_fill(~residue_mask, float("-inf"))

        attn_weights = F.softmax(raw_scores, dim=0)  # [num_residues]

        # Weighted sum
        h_protein = (attn_weights.unsqueeze(-1) * h_residues).sum(dim=0)  # [hidden_dim]

        return h_protein, attn_weights


class HybridProteinEncoder(nn.Module):
    """Hybrid ESM-C + EGNN protein encoder with hierarchical pooling.

    Combines per-residue sequence embeddings with atom-type one-hot
    encodings, processes through an EGNN structure encoder, and pools via
    hierarchical atom->residue->protein pooling.

    Supports two sequence embedding modes:
    - PLM mode (use_plm=True, default): Uses pre-computed ESM-C per-residue
      embeddings. Best for natural proteins with evolutionary context.
    - Vanilla mode (use_plm=False): Uses a learned amino acid embedding layer
      (20 AAs -> learned_aa_embedding_dim). Better for de novo proteins that
      may be out-of-distribution for protein language models.

    This replaces ProteInfer CNN in the ToxinNote architecture.

    Args:
        esmc_embedding_dim: Dimension of ESM-C embeddings (960 for esmc_300m)
        atom_type_dim: Number of atom types for one-hot encoding (37 for UNIFIED_ATOM37_ENCODING)
        egnn_hidden_dim: Hidden dimension for EGNN layers
        egnn_out_dim: Output dimension of EGNN
        egnn_n_layers: Number of E_GCL layers
        output_dim: Final protein embedding dimension
        use_plm: If True, use ESM-C embeddings. If False, use learned AA embeddings.
        num_amino_acids: Number of amino acid types for learned embedding (default 20)
        learned_aa_embedding_dim: Dimension of learned AA embedding when use_plm=False
    """

    def __init__(
        self,
        esmc_embedding_dim=960,
        atom_type_dim=37,
        egnn_hidden_dim=256,
        egnn_out_dim=256,
        egnn_n_layers=4,
        output_dim=256,
        use_plm=True,
        num_amino_acids=20,
        learned_aa_embedding_dim=128,
    ):
        super().__init__()

        self.use_plm = use_plm
        self.esmc_embedding_dim = esmc_embedding_dim
        self.atom_type_dim = atom_type_dim

        if use_plm:
            seq_embedding_dim = esmc_embedding_dim
        else:
            seq_embedding_dim = learned_aa_embedding_dim
            # Learned embedding: amino acid index -> dense vector
            # +1 for unknown/padding token at index 0
            self.aa_embedding = nn.Embedding(num_amino_acids + 1, learned_aa_embedding_dim, padding_idx=0)

        in_node_nf = seq_embedding_dim + atom_type_dim

        # EGNN structure encoder
        self.structure_encoder = StructureEncoder(
            in_node_nf=in_node_nf,
            hidden_nf=egnn_hidden_dim,
            out_node_nf=egnn_out_dim,
            n_layers=egnn_n_layers,
            residual=True,
            attention=False,
            normalize=False,
            tanh=False,
        )

        # Hierarchical pooling
        self.pooling = HierarchicalPooling(hidden_dim=egnn_out_dim)

        # Output projection to match ProtNote's expected protein embedding dim
        self.output_projection = nn.Linear(egnn_out_dim, output_dim)

    def get_embeddings(
        self,
        esmc_embeddings,
        atom_coords,
        atom_types,
        edge_index,
        atom_to_residue,
        residue_to_protein,
        num_residues_per_protein,
        num_proteins,
        residue_indices=None,
    ):
        """Compute protein embeddings from sequence embeddings and structure.

        All inputs use PyG-style batching: atoms from all proteins are concatenated
        into single tensors with index tensors tracking membership.

        Args:
            esmc_embeddings: Per-atom ESM-C embeddings (broadcast from residue) [N_atoms, esmc_embedding_dim].
                Used when use_plm=True.
            atom_coords: Atom 3D coordinates [N_atoms, 3]
            atom_types: Atom type one-hot vectors [N_atoms, atom_type_dim]
            edge_index: Edge indices [2, N_edges]
            atom_to_residue: Residue index for each atom [N_atoms] (global across batch)
            residue_to_protein: Protein index for each residue [N_residues_total]
            num_residues_per_protein: Number of residues per protein [num_proteins]
            num_proteins: Number of proteins in batch (int)
            residue_indices: Per-atom amino acid index [N_atoms] (1-indexed, 0=padding).
                Used when use_plm=False.

        Returns:
            protein_embeddings: [num_proteins, output_dim]
            attn_weights_list: List of per-protein attention weight tensors
        """
        # Build sequence embedding channel
        if self.use_plm:
            seq_emb = esmc_embeddings  # [N_atoms, esmc_embedding_dim]
        else:
            if residue_indices is None:
                raise ValueError("residue_indices required when use_plm=False. Ensure graph data includes per-atom amino acid indices.")
            seq_emb = self.aa_embedding(residue_indices)  # [N_atoms, learned_aa_embedding_dim]

        # Concatenate sequence embeddings with atom-type one-hot
        h = torch.cat([seq_emb, atom_types], dim=-1)

        # Run through EGNN (coordinates kept in float32 internally)
        h = self.structure_encoder(h, atom_coords, edge_index)  # [N_atoms, egnn_out_dim]

        # Per-protein hierarchical pooling
        protein_embeddings = []
        attn_weights_list = []

        # Compute residue offset for splitting
        total_residues = residue_to_protein.size(0)

        for i in range(num_proteins):
            # Get atom mask for this protein
            residue_mask = residue_to_protein == i
            residue_indices = torch.where(residue_mask)[0]
            num_res = num_residues_per_protein[i].item()

            if num_res == 0:
                # Fallback: empty protein
                protein_embeddings.append(h.new_zeros(self.structure_encoder.embedding_out.out_features))
                attn_weights_list.append(h.new_zeros(1))
                continue

            # Get atoms belonging to this protein's residues
            # atom_to_residue contains global residue indices
            min_res_idx = residue_indices[0].item()
            max_res_idx = residue_indices[-1].item()

            atom_mask = (atom_to_residue >= min_res_idx) & (atom_to_residue <= max_res_idx)
            h_atoms_i = h[atom_mask]
            # Remap atom_to_residue to local indices (0-based for this protein)
            local_atom_to_residue = atom_to_residue[atom_mask] - min_res_idx

            h_protein, attn_weights = self.pooling(h_atoms_i, local_atom_to_residue, num_res)
            protein_embeddings.append(h_protein)
            attn_weights_list.append(attn_weights)

        protein_embeddings = torch.stack(protein_embeddings, dim=0)  # [B, egnn_out_dim]

        # Project to output dimension
        protein_embeddings = self.output_projection(protein_embeddings)  # [B, output_dim]

        return protein_embeddings, attn_weights_list
