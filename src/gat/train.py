"""Training script for GAT model.

Usage:
    python -m src.gat.train --adjacency-dir /path/to/voronoi --sample-buildings /path/to/buildings.shp --sample-districts /path/to/districts.shp --output-root-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from .training import GATConfig, Trainer
from .models.gat import GAT
from .data import BuildingGraphDataset, BuildingDataset, DistrictDataset, split_dataset
from .utils import setup_logger


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train GAT model for building clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required path arguments
    parser.add_argument(
        '--adjacency-dir',
        type=str,
        required=True,
        help='Directory containing adjacency matrices (pkl files)'
    )
    parser.add_argument(
        '--sample-buildings',
        type=str,
        required=True,
        help='Path to building shapefile'
    )
    parser.add_argument(
        '--sample-districts',
        type=str,
        required=True,
        help='Path to district shapefile'
    )
    parser.add_argument(
        '--output-root-dir',
        type=str,
        required=True,
        help='Directory for all outputs (checkpoints, logs, embeddings)'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config YAML file (default: src/gat/training_config.yaml)'
    )

    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def main(args=None):
    """Main training function.

    Args:
        args: Optional argparse.Namespace. If None, will parse from sys.argv.
    """
    if args is None:
        args = parse_args()

    # Determine config path
    if args.config:
        config_path = Path(args.config)
    else:
        # Default config path relative to this file (src/gat/training_config.yaml)
        config_path = Path(__file__).parent / 'training_config.yaml'

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please specify a valid config file with --config or use the default training_config.yaml")
        sys.exit(1)

    # Load configuration from YAML
    resource_path = {
        'building_path': args.sample_buildings,
        'district_path': args.sample_districts,
        'adjacency_dir': args.adjacency_dir,
        'output_root_dir': args.output_root_dir,
    }
    config = GATConfig.from_yaml(config_path, resource_path=resource_path)
    print(f"Loaded configuration from {config_path}")

    if not Path(config.adjacency_dir).exists():
        print(f"Error: Data directory not found: {config.adjacency_dir}")
        sys.exit(1)

    if not Path(config.building_path).exists():
        print(f"Error: Building shapefile not found: {config.building_path}")
        sys.exit(1)

    if not Path(config.district_path).exists():
        print(f"Error: District shapefile not found: {config.district_path}")
        sys.exit(1)

    # Create output directory structure
    building_dataset = BuildingDataset(config.building_path)
    district_dataset = DistrictDataset(config.district_path)
    print(f"Output directory: {config.output_root_dir}")
    print(f"  - Checkpoints: {config.checkpoint_dir}")
    print(f"  - Logs: {config.log_dir}")
    print(f"  - Embeddings: {config.output_dir}")

    # Setup logger
    log_file = Path(config.log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(name='gat', log_file=log_file)

    logger.info("=" * 80)
    logger.info("GAT Training for Building Clustering")
    logger.info("=" * 80)
    logger.info("Configuration:\n%s", config)

    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = BuildingGraphDataset(
            adjacency_dir=config.adjacency_dir,
            district_dataset=district_dataset,
            building_dataset=building_dataset,
            dataset_dir=f"{config.output_root_dir}/dataset",
        )

        logger.info("Dataset loaded: %d districts", len(dataset))
        stats = dataset.get_statistics()
        logger.info("Dataset statistics: %s", stats)

        # Update config with actual number of features from dataset
        config.in_features = dataset.num_features
        logger.info("Auto-detected %d input features from dataset", config.in_features)

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
        in_features=dataset.num_features,
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
