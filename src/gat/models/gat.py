"""Graph Attention Network for building clustering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .gat_layer import GATConv
from ..utils.logger import get_logger
from ..utils.graph_utils import global_pool

logger = get_logger(__name__)


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network with ELU activation.
    
    Uses multi-head attention in hidden layers, single-head output.
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 8,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.6,
        negative_slope: float = 0.2,
        add_self_loops: bool = True
    ):
        """Initialize GAT with specified architecture."""
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        assert num_layers >= 2, "GAT requires at least 2 layers"

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(
                in_channels=in_features,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    negative_slope=negative_slope,
                    add_self_loops=add_self_loops
                )
            )

        self.convs.append(
            GATConv(
                in_channels=hidden_dim * num_heads,
                out_channels=num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            )
        )

        self.embedding_dim = hidden_dim * num_heads

        logger.debug(
            "GAT: %d layers, hidden=%d, heads=%d, classes=%d, dropout=%.2f",
            num_layers, hidden_dim, num_heads, num_classes, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through GAT layers.
        
        Returns logits (N, num_classes), or (logits, embeddings) if requested.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = x
        node_logits = self.convs[-1](x, edge_index, edge_attr)

        if return_embeddings:
            return node_logits, embeddings
        else:
            return node_logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract node embeddings from penultimate layer."""
        _, embeddings = self.forward(x, edge_index, edge_attr, return_embeddings=True)
        return embeddings

    def forward_inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and embeddings."""
        logits, embeddings = self.forward(
            x, edge_index, edge_attr, return_embeddings=True
        )
        return logits, embeddings

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract attention weights from specified layer."""
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv_idx, conv in enumerate(self.convs[:layer_idx + 1]):
            if conv_idx == layer_idx:
                x, (edge_index_out, attention) = conv(
                    x, edge_index, return_attention_weights=True
                )
                return edge_index_out, attention
            else:
                x = conv(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        raise ValueError(f"Layer index {layer_idx} out of range")

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.conv.reset_parameters()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n'
            f'  in_features={self.in_features},\n'
            f'  hidden_dim={self.hidden_dim},\n'
            f'  num_classes={self.num_classes},\n'
            f'  num_layers={self.num_layers},\n'
            f'  num_heads={self.num_heads},\n'
            f'  dropout={self.dropout},\n'
            f'  embedding_dim={self.embedding_dim}\n'
            f')'
        )

