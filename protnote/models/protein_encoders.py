"""
Protein Encoders.

StructureEncoder: Wraps E(n) Equivariant Graph Convolutional Layers (E_GCL)
for processing atom-level protein graphs.

Adapted from egnn/models/egnn_clean/egnn_clean.py (Satorras et al., 2021).
"""

import numpy as np
import torch
from torch import nn

from protnote.data.datasets import set_padding_to_sentinel
from protnote.utils.proteinfer import transfer_tf_weights_to_torch


class MaskedConv1D(torch.nn.Conv1d):
    def forward(self, x, sequence_lengths):
        """
        Correct for padding before and after. Can be redundant
        but reduces overhead of setting padding to sentiel in other contexts.
        """
        x = set_padding_to_sentinel(x, sequence_lengths, 0)
        x = super().forward(x)
        x = set_padding_to_sentinel(x, sequence_lengths, 0)
        return x


# ResNet-V2 https://arxiv.org/pdf/1602.07261v2.pdf


class Residual(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        dilation: int,
        bottleneck_factor: float,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        bottleneck_out_channels = int(np.floor(input_channels * bottleneck_factor))
        self.bn_activation_1 = torch.nn.Sequential(torch.nn.BatchNorm1d(input_channels, eps=0.001, momentum=0.01), activation())

        self.masked_conv1 = MaskedConv1D(
            in_channels=input_channels,
            out_channels=bottleneck_out_channels,
            padding="same",
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
        )
        self.bn_activation_2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(bottleneck_out_channels, eps=0.001, momentum=0.01),
            activation(),
        )

        self.masked_conv2 = MaskedConv1D(
            in_channels=bottleneck_out_channels,
            out_channels=input_channels,
            padding="same",
            kernel_size=1,
            stride=1,
            dilation=1,
        )

    def forward(self, x, sequence_lengths):
        out = self.bn_activation_1(x)
        out = self.masked_conv1(out, sequence_lengths)
        out = self.bn_activation_2(out)
        out = self.masked_conv2(out, sequence_lengths)
        out = out + x
        return out


class ProteInfer(torch.nn.Module):
    def __init__(
        self,
        num_labels: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        activation,
        dilation_base: int,
        num_resnet_blocks: int,
        bottleneck_factor: float,
    ):
        super().__init__()

        self.conv1 = MaskedConv1D(
            in_channels=input_channels,
            out_channels=output_channels,
            padding="same",
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
        )
        self.resnet_blocks = torch.nn.ModuleList()

        for i in range(num_resnet_blocks):
            self.resnet_blocks.append(
                Residual(
                    input_channels=output_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                    bottleneck_factor=bottleneck_factor,
                    activation=activation,
                )
            )

        self.output_layer = torch.nn.Linear(in_features=output_channels, out_features=num_labels)

    def get_embeddings(self, x, sequence_lengths):
        features = self.conv1(x, sequence_lengths)
        # Sequential doesn't work here because of multiple inputs
        for idx, resnet_block in enumerate(self.resnet_blocks):
            features = resnet_block(features, sequence_lengths)
        features = set_padding_to_sentinel(features, sequence_lengths, 0)
        features = torch.sum(features, dim=-1) / sequence_lengths.unsqueeze(-1)  # Average pooling
        return features

    def forward(self, x, sequence_lengths):
        features = self.get_embeddings(x, sequence_lengths)
        logits = self.output_layer(features)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str,
        num_labels: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        activation,
        dilation_base: int,
        num_resnet_blocks: int,
        bottleneck_factor: float,
    ):
        """
        Load a pretrained model from a path or url.
        """
        model = cls(
            num_labels,
            input_channels,
            output_channels,
            kernel_size,
            activation,
            dilation_base,
            num_resnet_blocks,
            bottleneck_factor,
        )
        transfer_tf_weights_to_torch(model, weights_path)

        return model


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
    """E(n) Equivariant Convolutional Layer.

    Updates node features h (invariant) and coordinates x (equivariant).
    """

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
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
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
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

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
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
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
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class StructureEncoder(nn.Module):
    """Wraps multiple E_GCL layers to encode atom-level protein graphs.

    Takes atom features h and 3D coordinates x, processes through N E_GCL layers,
    and returns updated invariant node features (coordinates are discarded).

    Args:
        in_node_nf: Input node feature dimension (e.g., 997 = 960 ESM-C + 37 atom-type)
        hidden_nf: Hidden feature dimension for E_GCL layers
        out_node_nf: Output node feature dimension
        n_layers: Number of E_GCL layers
        residual: Use residual connections in E_GCL
        attention: Use attention mechanism in E_GCL edge model
        normalize: Normalize coordinate messages
        tanh: Apply tanh to coordinate updates
    """

    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        # Input projection: in_node_nf -> hidden_nf
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)

        # Stack of E_GCL layers
        self.gcl_layers = nn.ModuleList(
            [
                E_GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    act_fn=nn.SiLU(),
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection: hidden_nf -> out_node_nf
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, h, x, edge_index, edge_attr=None):
        """Process atom graph through EGNN layers.

        Args:
            h: Node features [N_atoms, in_node_nf]
            x: Node coordinates [N_atoms, 3]
            edge_index: Edge indices [2, N_edges]
            edge_attr: Optional edge features [N_edges, edge_dim]

        Returns:
            h_out: Updated invariant node features [N_atoms, out_node_nf]
        """
        # Keep coordinates in float32 for numerical stability
        x = x.float()

        h = self.embedding_in(h)
        for gcl in self.gcl_layers:
            h, x, _ = gcl(h, edge_index, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h
