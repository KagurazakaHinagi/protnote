"""
E(3) Equivariant GNN for structural protein encoding.
Ported from StrucToxNet (Jiao et al.) for use as ProtNote's protein branch.

Updated to support atom-level graphs with ESM-C embeddings.
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    import torch_scatter
except ImportError:
    torch_scatter = None


# Default atom type dimension for one-hot encoding (covers common protein elements)
ATOM_TYPE_DIM = 37


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """E(3) Equivariant Graph Convolutional Layer."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        # edge_mlp input: [source, target, radial, edge_attr] -> 2*hidden_nf + 1 + edges_in_d
        edge_in_dim = 2 * hidden_nf + 1 + (edges_in_d if edges_in_d else 0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = out.to(torch.float32)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index[0], edge_index[1]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        agg = agg.to(torch.float32)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index[0], edge_index[1]
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class StructuralProteinEncoder(nn.Module):
    """
    Structural-Sequential encoder: PLM node features + EGNN -> graph-level embedding.
    Interface compatible with ProtNote's sequence_encoder.

    Supports two input formats:
    - Legacy PyG Batch format: get_embeddings(batch) with .x, .plm, .edge_index, .edge_s, .batch
    - Atom-level dict format: get_embeddings(**graph_data) with ESM-C + atom-type features
    """

    def __init__(
        self,
        plm_embedding_dim=1069,
        protein_embedding_dim=1100,
        hidden_nf=256,
        num_layers=3,
        in_edge_nf=16,
        dropout=0.5,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        # New params for atom-level input
        atom_type_dim=ATOM_TYPE_DIM,
        use_atom_level=False,
    ):
        super().__init__()
        if torch_scatter is None:
            raise ImportError("StructuralProteinEncoder requires torch_scatter. Install with: pip install torch_scatter")

        self.plm_embedding_dim = plm_embedding_dim
        self.protein_embedding_dim = protein_embedding_dim
        self.hidden_nf = hidden_nf
        self.num_layers = num_layers
        self.use_atom_level = use_atom_level
        self.atom_type_dim = atom_type_dim

        # Input dimension depends on mode
        if use_atom_level:
            # Atom-level: ESM-C + atom-type one-hot
            in_node_dim = plm_embedding_dim + atom_type_dim
            self.in_edge_nf = 0  # No explicit edge features for atom-level
        else:
            # Legacy residue-level
            in_node_dim = plm_embedding_dim
            self.in_edge_nf = in_edge_nf

        self.W_v = nn.Linear(in_node_dim, hidden_nf, bias=True)
        self.gcl_layers = nn.ModuleList(
            [
                E_GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=self.in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                )
                for _ in range(num_layers)
            ]
        )
        # Map pooled hidden_nf -> protein_embedding_dim (e.g. 1100 for ProtNote W_p)
        # Use LayerNorm instead of BatchNorm1d to support batch_size=1
        self.projection = nn.Sequential(
            nn.Linear(hidden_nf, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, protein_embedding_dim),
        )

    def forward(self, batch):
        """
        Legacy forward for PyG Batch format.
        batch: PyG Batch with .x (coords), .plm (node PLM embeddings), .edge_index, .edge_s, .batch
        """
        x = batch.x
        h = batch.plm
        h = self.W_v(h)
        for gcl in self.gcl_layers:
            h, x, _ = gcl(h, batch.edge_index, x, edge_attr=batch.edge_s)
        out = torch_scatter.scatter_max(h, batch.batch, dim=0)[0].float()
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda", enabled=False):
            return self.projection(out)

    def forward_atom_level(
        self,
        esmc_embeddings,
        atom_coords,
        atom_types,
        edge_index,
        atom_to_protein,
        num_proteins,
        **kwargs,
    ):
        """
        Forward pass for atom-level graph data.

        Args:
            esmc_embeddings: Per-atom ESM-C embeddings [N_atoms, esmc_dim]
            atom_coords: Atom 3D coordinates [N_atoms, 3]
            atom_types: Atom type indices [N_atoms] or one-hot [N_atoms, atom_type_dim]
            edge_index: Edge indices [2, N_edges]
            atom_to_protein: Protein index for each atom [N_atoms]
            num_proteins: Number of proteins in batch

        Returns:
            protein_embeddings: [num_proteins, protein_embedding_dim]
        """
        # Convert atom_types to one-hot if needed
        if atom_types.dim() == 1:
            atom_types_onehot = F.one_hot(atom_types, num_classes=self.atom_type_dim).float()
        else:
            atom_types_onehot = atom_types.float()

        # Concatenate ESM-C with atom-type one-hot
        h = torch.cat([esmc_embeddings, atom_types_onehot], dim=-1)
        x = atom_coords.float()

        # Project to hidden dim
        h = self.W_v(h)

        # Run through EGNN layers (no edge attributes for atom-level)
        for gcl in self.gcl_layers:
            h, x, _ = gcl(h, edge_index, x, edge_attr=None)

        # Global max pooling per protein
        out = torch_scatter.scatter_max(h, atom_to_protein, dim=0, dim_size=num_proteins)[0].float()

        # Guard against -inf from scatter_max (proteins with no atoms get -inf fill)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        # Run projection in float32 to avoid autocast float16 overflow in BatchNorm
        with torch.amp.autocast("cuda", enabled=False):
            return self.projection(out)

    def get_embeddings(self, batch_or_data=None, **kwargs):
        """
        Unified interface supporting both input formats.

        For legacy PyG Batch:
            get_embeddings(batch) where batch has .x, .plm, .edge_index, .edge_s, .batch

        For atom-level dict:
            get_embeddings(esmc_embeddings=..., atom_coords=..., atom_types=...,
                          edge_index=..., atom_to_protein=..., num_proteins=...)
        """
        # Check if called with keyword args (atom-level format)
        if batch_or_data is None and kwargs:
            return self.forward_atom_level(**kwargs)

        # Check if batch_or_data is a dict (atom-level format)
        if isinstance(batch_or_data, dict):
            return self.forward_atom_level(**batch_or_data)

        # Otherwise assume PyG Batch (legacy format)
        return self.forward(batch_or_data)
