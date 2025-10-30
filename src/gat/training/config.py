"""Configuration for GAT training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import torch
import yaml


@dataclass
class GATConfig:
    """
    Configuration for GAT model and training.

    Follows pytorch-GAT default settings for small graphs (Cora-like),
    adapted for building clustering task.
    """

    hidden_dim: int = 64  # Hidden dimension per head
    num_classes: int = 8  # Number of building categories (8 classes)
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
    lambda_smooth: float = 0.5  # Spatial smoothness loss weight
    smooth_temperature: float = 1.0  # Temperature for smoothness loss softmax

    # Data parameters
    batch_size: int = 1024  # Nodes per batch for NeighborLoader
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10])  # Neighbor sampling
    node_threshold: int = 2000  # Use sampling for graphs > this size
    train_ratio: float = 0.8  # Train/val split ratio
    num_workers: int = 0  # DataLoader workers (0 for debugging, 4+ for speed)

    # resource paths
    adjacency_dir: str = ""
    building_path: str = ""
    district_path: str = ""
    output_root_dir: str = ""
    model_identifier: str = "default"  # Model version identifier

    # subdirectories Path
    checkpoint_dir: str = "models"  # Model checkpoints
    log_dir: str = "runs"  # TensorBoard logs
    output_dir: str = "output"  # Output embeddings

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
        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

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
        model_dict = {
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'negative_slope': self.negative_slope,
            'add_self_loops': self.add_self_loops,
        }

        return {
            'model': model_dict,
            'training': {
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'num_neighbors': self.num_neighbors,
                'train_ratio': self.train_ratio,
                'lambda_smooth': self.lambda_smooth,
                'smooth_temperature': self.smooth_temperature,
            },
            'device': self.device,
            'seed': self.seed,
            'model_identifier': self.model_identifier,
        }

    def __repr__(self):
        return (
            f"GATConfig(\n"
            f"  Model: {self.num_layers} layers, hidden={self.hidden_dim}, heads={self.num_heads}\n"
            f"  Training: lr={self.lr}, epochs={self.epochs}, batch_size={self.batch_size}\n"
            f"  Device: {self.device}\n"
            f"  Data: {self.adjacency_dir}\n"
            f")"
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path, resource_path: Dict[str, Any]) -> 'GATConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            resource_path: Dictionary containing resource paths (adjacency_dir, building_shapefile, etc.)

        Returns:
            GATConfig instance
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Extract parameters from nested structure
        model_params = config_dict.get('model', {})
        training_params = config_dict.get('training', {})
        data_params = config_dict.get('data', {})
        logging_params = config_dict.get('logging', {})

        # Build flat parameter dict
        params = {
            'hidden_dim': model_params.get('hidden_dim', 64),
            'num_classes': data_params.get('num_classes', 3),
            'num_layers': model_params.get('num_layers', 3),
            'num_heads': model_params.get('num_heads', 8),
            'dropout': model_params.get('dropout', 0.6),
            'negative_slope': model_params.get('negative_slope', 0.2),
            'add_self_loops': model_params.get('add_self_loops', True),

            # Training parameters
            'lr': training_params.get('lr', 5e-3),
            'weight_decay': training_params.get('weight_decay', 5e-4),
            'epochs': training_params.get('epochs', 200),
            'patience': training_params.get('patience', 100),
            'min_delta': training_params.get('min_delta', 1e-4),
            'batch_size': training_params.get('batch_size', 1024),
            'num_neighbors': training_params.get('num_neighbors', [15, 10]),
            'node_threshold': training_params.get('node_threshold', 2000),
            'train_ratio': training_params.get('train_ratio', 0.8),
            'num_workers': training_params.get('num_workers', 0),
            'use_amp': training_params.get('use_amp', False),
            'gradient_accumulation_steps': training_params.get('gradient_accumulation_steps', 1),
            'lambda_smooth': training_params.get('lambda_smooth', 0.5),
            'smooth_temperature': training_params.get('smooth_temperature', 1.0),

            # Data parameters
            'district_ids': data_params.get('district_ids', []),

            # Logging parameters
            'log_interval': logging_params.get('log_interval', 10),
            'checkpoint_interval': logging_params.get('checkpoint_interval', 50),
            'enable_tensorboard': logging_params.get('enable_tensorboard', True),

            # Other parameters
            'seed': config_dict.get('seed', 42),
            'device': config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        }

        # Add resource paths directly
        params.update(resource_path)

        # Construct output subdirectories based on output_dir
        params['checkpoint_dir'] = f"{params['output_root_dir']}/checkpoints"
        params['log_dir'] = f"{params['output_root_dir']}/logs"
        params['output_dir'] = f"{params['output_root_dir']}/output_{params['model_identifier']}"
        return cls(**params)
