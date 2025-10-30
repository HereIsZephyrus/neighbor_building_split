"""GAT layer implementation using PyTorch Geometric.

This module wraps PyG's GATConv for consistency with the rest of the codebase.
We use PyG's efficient implementation which follows the original GAT paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PyGGATConv
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger()


class GATConv(nn.Module):
    """
    Graph Attention Convolutional Layer.

    Implements multi-head attention mechanism following Veličković et al. (2018).
    This is a wrapper around PyG's GATConv for easier customization.

    Attention mechanism:
        α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
        h'_i = σ(Σ_j α_ij W h_j)

    where:
        - W is the learnable weight matrix
        - a is the attention mechanism weights
        - || denotes concatenation
        - σ is an activation function (ELU in our case)
    """

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
        """
        Initialize GAT convolutional layer.

        Args:
            in_channels: Size of input features
            out_channels: Size of output features per head
            heads: Number of attention heads
            concat: If True, concatenate outputs from all heads.
                   If False, average them.
            negative_slope: LeakyReLU negative slope
            dropout: Dropout probability for attention weights
            add_self_loops: Whether to add self-loops to the graph
            bias: Whether to use bias
            **kwargs: Additional arguments for GATConv
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Use PyG's GATConv implementation
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

        # Output dimension
        if concat:
            self.output_dim = out_channels * heads
        else:
            self.output_dim = out_channels

        logger.debug(
            f"Created GATConv layer: in={in_channels}, out={out_channels}, "
            f"heads={heads}, concat={concat}, output_dim={self.output_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Forward pass.

        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_attr: Optional edge attributes (E,) - currently not used by GATConv
            return_attention_weights: Whether to return attention weights

        Returns:
            out: Output node features (N, output_dim)
            attention_weights: Optional (edge_index, attention) tuple
        """
        # PyG's GATConv handles the attention mechanism internally
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

