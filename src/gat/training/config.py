"""Configuration for GAT training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class GATConfig:
    """
    Configuration for GAT model and training.

    Follows pytorch-GAT default settings for small graphs (Cora-like),
    adapted for building clustering task.
    """

    # Model architecture
    in_features: int = 12  # Number of building features
    hidden_dim: int = 64  # Hidden dimension per head
    num_classes: int = 3  # Number of building categories (will be auto-detected)
    num_layers: int = 3  # Number of GAT layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.6  # Dropout rate (as in pytorch-GAT)
    negative_slope: float = 0.2  # LeakyReLU slope for attention
    add_self_loops: bool = True  # Add self-loops to graphs

    # Training parameters
    lr: float = 5e-3  # Learning rate (as in pytorch-GAT)
    weight_decay: float = 5e-4  # L2 regularization (as in pytorch-GAT)
    epochs: int = 200  # Maximum number of epochs
    patience: int = 100  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping

    # Data parameters
    batch_size: int = 1024  # Nodes per batch for NeighborLoader
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10])  # Neighbor sampling
    node_threshold: int = 2000  # Use sampling for graphs > this size
    train_ratio: float = 0.8  # Train/val split ratio
    num_workers: int = 0  # DataLoader workers (0 for debugging, 4+ for speed)

    # Paths
    data_dir: str = "output/voronoi"  # Directory with adjacency matrices
    building_shapefile: str = ""  # Path to building shapefile (required)
    checkpoint_dir: str = "models/gat"  # Model checkpoints
    log_dir: str = "runs/gat"  # TensorBoard logs
    output_dir: str = "output/gat"  # Output embeddings

    # Device and optimization
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = False  # Use automatic mixed precision (optional for 8GB GPU)
    gradient_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch

    # Logging and checkpointing
    log_interval: int = 10  # Log every N epochs
    checkpoint_interval: int = 50  # Save checkpoint every N epochs
    enable_tensorboard: bool = True  # Enable TensorBoard logging

    # Random seed
    seed: int = 42

    # District IDs (empty = auto-detect from data_dir)
    district_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation and path conversion."""
        # Convert paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.output_dir = Path(self.output_dir)

        if self.building_shapefile:
            self.building_shapefile = Path(self.building_shapefile)

        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers >= 2, "num_layers must be at least 2"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.lr > 0, "lr must be positive"
        assert 0 < self.train_ratio < 1, "train_ratio must be in (0, 1)"
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': {
                'in_features': self.in_features,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
                'negative_slope': self.negative_slope,
                'add_self_loops': self.add_self_loops,
            },
            'training': {
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'num_neighbors': self.num_neighbors,
                'train_ratio': self.train_ratio,
            },
            'device': self.device,
            'seed': self.seed,
        }

    def __repr__(self):
        return (
            f"GATConfig(\n"
            f"  Model: {self.num_layers} layers, hidden={self.hidden_dim}, heads={self.num_heads}\n"
            f"  Training: lr={self.lr}, epochs={self.epochs}, batch_size={self.batch_size}\n"
            f"  Device: {self.device}\n"
            f"  Data: {self.data_dir}\n"
            f")"
        )

