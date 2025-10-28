"""Main entry point for GAT training.

Usage:
    python -m src.gat --data-dir output/voronoi --building-shapefile path/to/buildings.shp
    python -m src.gat --config config.json
    python -m src.gat --resume models/gat/checkpoint_epoch_100.pth
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import torch

from .training.config import GATConfig
from .training.trainer import Trainer
from .models.gat import GAT
from .data.dataset import BuildingGraphDataset
from .data.data_utils import split_dataset
from .utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GAT model for building clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='output/voronoi',
        help='Directory containing adjacency matrices (pkl files)'
    )
    parser.add_argument(
        '--building-shapefile',
        type=str,
        required=False,
        help='Path to building shapefile (required if not in config)'
    )
    parser.add_argument(
        '--district-ids',
        type=int,
        nargs='+',
        default=None,
        help='List of district IDs to train on (default: auto-detect)'
    )

    # Model arguments
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension per attention head'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of GAT layers'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.6,
        help='Dropout rate'
    )

    # Training arguments
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size (nodes per batch for NeighborLoader)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=100,
        help='Early stopping patience'
    )

    # Paths
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/gat',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='runs/gat',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/gat',
        help='Directory for output embeddings'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file (overrides other arguments)'
    )

    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )

    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> GATConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return GATConfig(**config_dict)


def auto_detect_district_ids(data_dir: Path) -> list:
    """Auto-detect district IDs from adjacency matrix files."""
    district_ids = []

    for pkl_file in data_dir.glob('district_*_adjacency.pkl'):
        try:
            # Extract district ID from filename
            district_id = int(pkl_file.stem.split('_')[1])
            district_ids.append(district_id)
        except (IndexError, ValueError):
            continue

    district_ids.sort()
    return district_ids


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Create config from command line arguments
        config = GATConfig(
            data_dir=args.data_dir,
            building_shapefile=args.building_shapefile or '',
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            enable_tensorboard=not args.no_tensorboard,
            district_ids=args.district_ids or []
        )

    # Validate required paths
    building_shapefile_path = Path(config.building_shapefile) if config.building_shapefile else None
    if not building_shapefile_path or not building_shapefile_path.exists():
        print("Error: Building shapefile path is required and must exist!")
        print("Use --building-shapefile to specify the path.")
        sys.exit(1)

    if not config.data_dir.exists():
        print(f"Error: Data directory not found: {config.data_dir}")
        sys.exit(1)

    # Setup logger
    log_file = config.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(name='gat', log_file=log_file)

    logger.info("=" * 80)
    logger.info("GAT Training for Building Clustering")
    logger.info("=" * 80)
    logger.info("Configuration:\n%s", config)

    # Auto-detect district IDs if not specified
    if not config.district_ids:
        config.district_ids = auto_detect_district_ids(config.data_dir)
        logger.info("Auto-detected %d districts: %s", len(config.district_ids), config.district_ids)

    if len(config.district_ids) == 0:
        logger.error("No districts found! Please check data directory.")
        sys.exit(1)

    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = BuildingGraphDataset(
            root=str(config.data_dir),
            district_ids=config.district_ids,
            building_shapefile_path=str(config.building_shapefile),
            normalize_features=True
        )

        logger.info("Dataset loaded: %d districts", len(dataset))
        stats = dataset.get_statistics()
        logger.info("Dataset statistics: %s", stats)

        # Update num_classes in config
        config.num_classes = dataset.get_num_classes()
        logger.info("Number of classes: %d", config.num_classes)

    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc, exc_info=True)
        sys.exit(1)

    # Split into train/val
    logger.info("Splitting dataset: %.0f%% train, %.0f%% val", config.train_ratio * 100, (1-config.train_ratio) * 100)
    data_list = [dataset.get(i) for i in range(len(dataset))]
    train_data, val_data = split_dataset(data_list, train_ratio=config.train_ratio, random_seed=config.seed)

    logger.info("Train: %d districts, Val: %d districts", len(train_data), len(val_data))

    # Initialize model
    logger.info("Initializing model...")
    model = GAT(
        in_features=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        negative_slope=config.negative_slope,
        add_self_loops=config.add_self_loops
    )

    logger.info("Model:\n%s", model)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_data_list=train_data,
        val_data_list=val_data
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.resume_from_checkpoint(Path(args.resume))

    # Train
    try:
        history = trainer.train()

        logger.info("Training completed successfully!")
        logger.info("Best validation accuracy: %.4f", max(history.get('val_acc', [0])))

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

