"""Configuration for GAT training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import torch
import yaml
from datetime import datetime


@dataclass
class GATConfig:
    """
    Configuration for GAT model and training.

    Follows pytorch-GAT default settings for small graphs (Cora-like),
    adapted for building clustering task.
    """

    hidden_dim: int = 64  # Hidden dimension per head
    num_classes: int = 9  # Number of building categories (9 classes)
    num_layers: int = 3  # Number of GAT layers
    num_heads: int = 8  # Number of attention heads
    dropout: float = 0.6  # Dropout rate (as in pytorch-GAT)
    negative_slope: float = 0.2  # LeakyReLU slope for attention
    add_self_loops: bool = True  # Add self-loops to graphs

    # Training parameters
    lr: float = 5e-3  # Learning rate (as in pytorch-GAT)
    weight_decay: float = 5e-4  # L2 regularization (as in pytorch-GAT)
    epochs: int = 1000  # Maximum number of epochs
    patience: int = 100  # Early stopping patience
    min_delta: float = 0.01  # Minimum improvement for early stopping
    val_interval: int = 10  # Validate every N epochs
    lambda_smooth: float = 0.5  # Spatial smoothness loss weight
    smooth_temperature: float = 1.0  # Temperature for smoothness loss softmax

    # Data parameters
    batch_size: int = 1024  # Nodes per batch for NeighborLoader
    num_neighbors: List[int] = field(default_factory=lambda: [15, 10])  # Neighbor sampling
    node_threshold: int = 2000  # Use sampling for graphs > this size
    num_workers: int = 0  # DataLoader workers (0 for debugging, 4+ for speed)
    k_fold: int = 8  # Number of folds for cross-validation

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
    config_backup_dir: str = "training_configs"  # Training config backups
    config_dict_dir: str = "config_dicts"  # Training config dicts
    # Device and optimization
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = False  # Use automatic mixed precision (optional for 8GB GPU)
    gradient_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch

    # Logging and checkpointing
    log_interval: int = 10  # Log every N epochs
    checkpoint_interval: int = 50  # Save checkpoint every N epochs
    enable_checkpoint_saving: bool = True  # Enable periodic checkpoint saving (disable for HPC tuning)
    enable_tensorboard: bool = True  # Enable TensorBoard logging

    # Visualization
    enable_visualization: bool = True  # Enable district visualization
    max_visualize_districts: int = 9  # Maximum districts to visualize

    # Random seed
    seed: int = 42

    # Loss function parameters (for handling class imbalance)
    class_weight_smoothing: str = 'sqrt'  # Class weight smoothing: 'sqrt', 'inverse', or 'log'
    use_focal_loss: bool = False  # Whether to use Focal Loss instead of CrossEntropy
    focal_gamma: float = 2.0  # Focal loss gamma parameter (higher = more focus on hard examples)
    label_smoothing: float = 0.0  # Label smoothing factor (0.0-1.0, typically 0.1)

    # Spectral Clustering Configuration (used during inference for spatial smoothing)
    spectral_embedding_weight: float = 0.3  # Weight for GAT embedding similarity
    spectral_feature_weight: float = 0.5  # Weight for morphological feature similarity
    spectral_distance_weight: float = 0.2  # Weight for spatial distance affinity
    spectral_distance_scale: float = 100.0  # Distance-to-affinity conversion scale (meters)
    spectral_oversample_factor: float = 1.0  # Cluster oversampling factor
    spectral_min_component_size: int = 3  # Minimum component size for spectral clustering
    spectral_min_cluster_size: int = 5  # Minimum cluster size (smaller clusters revert to GAT)
    spectral_max_hops: int = 2  # Maximum graph hops for clustering
    spectral_use_confidence_weighted_voting: bool = False  # Use confidence-weighted voting

    def __post_init__(self):
        """Post-initialization validation and path conversion."""
        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate GAT parameters
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers >= 2, "num_layers must be at least 2"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.lr > 0, "lr must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.k_fold >= 2, "k_fold must be at least 2"

        # Validate spectral clustering parameters
        weight_sum = self.spectral_embedding_weight + self.spectral_feature_weight + self.spectral_distance_weight
        assert abs(weight_sum - 1.0) < 1e-6, f"Spectral clustering weights must sum to 1.0, got {weight_sum:.4f}"
        assert self.spectral_distance_scale > 0, "spectral_distance_scale must be positive"

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
            'hidden_dim': model_params.get('hidden_dim'),
            'num_classes': data_params.get('num_classes'),
            'num_layers': model_params.get('num_layers'),
            'num_heads': model_params.get('num_heads'),
            'dropout': model_params.get('dropout'),
            'negative_slope': model_params.get('negative_slope'),
            'add_self_loops': model_params.get('add_self_loops'),

            # Training parameters
            'lr': training_params.get('lr'),
            'weight_decay': training_params.get('weight_decay'),
            'epochs': training_params.get('epochs'),
            'patience': training_params.get('patience'),
            'min_delta': training_params.get('min_delta'),
            'val_interval': training_params.get('val_interval'),
            'batch_size': training_params.get('batch_size'),
            'num_neighbors': training_params.get('num_neighbors'),
            'node_threshold': training_params.get('node_threshold'),
            'k_fold': training_params.get('k_fold'),
            'num_workers': training_params.get('num_workers'),
            'use_amp': training_params.get('use_amp'),
            'gradient_accumulation_steps': training_params.get('gradient_accumulation_steps'),
            'lambda_smooth': training_params.get('lambda_smooth'),
            'smooth_temperature': training_params.get('smooth_temperature'),

            # Logging parameters
            'log_interval': logging_params.get('log_interval'),
            'checkpoint_interval': logging_params.get('checkpoint_interval'),
            'enable_checkpoint_saving': logging_params.get('enable_checkpoint_saving'),
            'enable_tensorboard': logging_params.get('enable_tensorboard'),

            # Other parameters
            'seed': config_dict.get('seed'),
            'device': config_dict.get('device'),

            # Loss function parameters
            'class_weight_smoothing': data_params.get('class_weight_smoothing'),
            'use_focal_loss': config_dict.get('focal_loss', {}).get('enabled'),
            'focal_gamma': config_dict.get('focal_loss', {}).get('gamma'),
            'label_smoothing': config_dict.get('strategy', {}).get('label_smoothing'),
        }

        # Spectral Clustering parameters (for inference stage)
        spectral_params = config_dict.get('spectral_clustering', {})
        params.update({
            'spectral_embedding_weight': spectral_params.get('embedding_weight'),
            'spectral_feature_weight': spectral_params.get('feature_weight'),
            'spectral_distance_weight': spectral_params.get('distance_weight'),
            'spectral_distance_scale': spectral_params.get('distance_scale'),
            'spectral_oversample_factor': spectral_params.get('oversample_factor'),
            'spectral_min_component_size': spectral_params.get('min_component_size'),
            'spectral_min_cluster_size': spectral_params.get('min_cluster_size'),
            'spectral_max_hops': spectral_params.get('max_hops'),
            'spectral_use_confidence_weighted_voting': spectral_params.get('use_confidence_weighted_voting'),
        })

        # Filter out None values - let dataclass use its defaults
        params = {k: v for k, v in params.items() if v is not None}

        # Add resource paths directly
        params.update(resource_path)

        # Construct output subdirectories based on output_dir
        params['log_dir'] = f"{params['output_root_dir']}/logs/{params['model_identifier']}"
        params['model_identifier'] = f"{params['model_identifier']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        params['checkpoint_dir'] = f"{params['output_root_dir']}/checkpoints/{params['model_identifier']}"
        params['output_dir'] = f"{params['output_root_dir']}/output/{params['model_identifier']}"
        params['config_backup_dir'] = f"{params['output_root_dir']}/training_configs"
        params['config_dict_dir'] = f"{params['output_root_dir']}/config_dicts"
        return cls(**params)
