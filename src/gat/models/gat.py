"""Graph Attention Network (GAT) model.

Implementation following pytorch-GAT (Gordicaleksa's implementation)
with architecture adapted for building clustering task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .gat_layer import GATConv
from ..utils.logger import get_logger

logger = get_logger()


class GAT(nn.Module):
    """
    Graph Attention Network for node classification.
    
    Architecture (similar to pytorch-GAT for Cora):
        - Layer 1: in_features → hidden_dim (heads=num_heads, concat)
        - Layer 2: hidden_dim*heads → hidden_dim (heads=num_heads, concat)
        - Layer 3: hidden_dim*heads → num_classes (heads=1, average)
    
    Activation: ELU (as in original paper)
    Dropout: Applied to both input features and attention weights
    """
    
    def __init__(
        self,
        in_features: int = 12,
        hidden_dim: int = 64,
        num_classes: int = 3,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.6,
        negative_slope: float = 0.2,
        add_self_loops: bool = True
    ):
        """
        Initialize GAT model.
        
        Args:
            in_features: Number of input features per node
            hidden_dim: Hidden dimension per attention head
            num_classes: Number of output classes
            num_layers: Number of GAT layers (default: 3)
            num_heads: Number of attention heads in hidden layers
            dropout: Dropout rate (applied to features and attention)
            negative_slope: LeakyReLU negative slope for attention
            add_self_loops: Whether to add self-loops
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert num_layers >= 2, "GAT requires at least 2 layers"
        
        # Create GAT layers
        self.convs = nn.ModuleList()
        
        # First layer: in_features → hidden_dim
        self.convs.append(
            GATConv(
                in_channels=in_features,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,  # Concatenate attention heads
                dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            )
        )
        
        # Hidden layers: (hidden_dim * num_heads) → hidden_dim
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
        
        # Final layer: (hidden_dim * num_heads) → num_classes
        # Use single head and average (as in pytorch-GAT)
        self.convs.append(
            GATConv(
                in_channels=hidden_dim * num_heads,
                out_channels=num_classes,
                heads=1,
                concat=False,  # Average the single head
                dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            )
        )
        
        # Embedding dimension (for downstream clustering)
        self.embedding_dim = hidden_dim * num_heads
        
        logger.info(
            "Initialized GAT: layers=%d, hidden_dim=%d, heads=%d, classes=%d, dropout=%.2f, embedding_dim=%d",
            num_layers, hidden_dim, num_heads, num_classes, dropout, self.embedding_dim
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            edge_attr: Optional edge attributes (E,) - not used currently
            return_embeddings: If True, return penultimate layer embeddings
            
        Returns:
            If return_embeddings=False:
                logits: Output class logits (N, num_classes)
            If return_embeddings=True:
                (logits, embeddings) where embeddings is (N, embedding_dim)
        """
        # Input dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Forward through all layers except the last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)  # ELU activation as in original paper
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Save embeddings before final layer
        embeddings = x
        
        # Final layer (no activation, no dropout after)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        if return_embeddings:
            return x, embeddings
        else:
            return x
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract node embeddings from penultimate layer.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            edge_attr: Optional edge attributes (E,)
            
        Returns:
            embeddings: Node embeddings (N, embedding_dim)
        """
        _, embeddings = self.forward(x, edge_index, edge_attr, return_embeddings=True)
        return embeddings
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights from a specific layer.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)
            layer_idx: Which layer to extract attention from
            
        Returns:
            edge_index: Edge indices with self-loops (2, E')
            attention_weights: Attention weights (E', num_heads)
        """
        # Forward through layers up to layer_idx
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for conv_idx, conv in enumerate(self.convs[:layer_idx + 1]):
            if conv_idx == layer_idx:
                # Return attention weights from this layer
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

