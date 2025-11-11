"""GAT convolutional layer using PyTorch Geometric."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PyGGATConv
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GATConv(nn.Module):
    """Multi-head graph attention convolution layer (wrapper for PyG's GATConv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs
    ):
        """Initialize GAT convolution layer."""
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.conv = PyGGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            **kwargs
        )

        if concat:
            self.output_dim = out_channels * heads
        else:
            self.output_dim = out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """Forward pass with multi-head attention."""
        if return_attention_weights:
            out, (edge_index_with_self_loops, attention_weights) = self.conv(
                x, edge_index, return_attention_weights=True
            )
            return out, (edge_index_with_self_loops, attention_weights)
        else:
            out = self.conv(x, edge_index)
            return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'heads={self.heads}, '
            f'concat={self.concat})'
        )

