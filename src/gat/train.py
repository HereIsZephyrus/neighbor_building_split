"""Training script for GAT model.

Usage:
    python -m src.gat.train --adjacency-dir /path/to/voronoi --sample-buildings /path/to/buildings.shp --sample-districts /path/to/districts.shp --output-root-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil
import yaml

from .training import GATConfig
from .data import BuildingGraphDataset, BuildingDataset, DistrictDataset
from .utils import setup_logger
from .train_cv import train_cross_validation_mpi, train_cross_validation_sequential
from .train_final import train_final_model

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

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

    # Training mode
    parser.add_argument(
        '--mode',
        type=str,
        default='cv',
        choices=['cv', 'final'],
        help='Training mode: "cv" for cross-validation (hyperparameter tuning), "final" for full training on all data'
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
        'model_identifier': getattr(args, 'model_identifier', 'default'),
    }
    config = GATConfig.from_yaml(config_path, resource_path=resource_path)
    print(f"Loaded configuration from {config_path}")
    print(f"Model identifier: {config.model_identifier}")

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
    # mkdir for output_root_dir
    Path(config.output_root_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.config_backup_dir).mkdir(parents=True, exist_ok=True)
    Path(config.config_dict_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {config.output_root_dir}")
    print(f"  - Checkpoints: {config.checkpoint_dir}")
    print(f"  - Logs: {config.log_dir}")
    print(f"  - Outputs: {config.output_dir}")
    print(f"  - Config backups: {config.config_backup_dir}")
    print(f"  - Config dicts: {config.config_dict_dir}")
    # Save training config to output directory for reference (MPI-safe: only rank 0)
    # Get rank before saving to avoid multiple processes writing simultaneously
    # Check if we should use MPI (available and more than 1 process)
    use_mpi = False
    current_rank = 0
    world_size = 1
    
    if MPI_AVAILABLE:
        try:
            comm_temp = MPI.COMM_WORLD
            current_rank = comm_temp.Get_rank()
            world_size = comm_temp.Get_size()
            # Only use MPI if we have more than 1 process
            use_mpi = (world_size > 1)
            if not use_mpi and current_rank == 0:
                print(f"MPI available but only 1 process detected, running in sequential mode")
        except:
            pass

    if current_rank == 0:
        config_backup_path = Path(config.config_backup_dir) / f'{config.model_identifier}.yaml'
        try:
            shutil.copy(config_path, config_backup_path)
            print(f"Training config saved to: {config_backup_path}")

            # Also save the full config as a dict for easier inspection
            config_dict_path = Path(config.config_dict_dir) / f'{config.model_identifier}.yaml'
            with open(config_dict_path, 'w', encoding='utf-8') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
            print(f"Config dict saved to: {config_dict_path}")
        except Exception as exc:
            print(f"Warning: Failed to save config backup: {exc}")

    # Setup logger (MPI-safe: each rank gets its own log file)
    if use_mpi:
        rank = current_rank
    else:
        rank = 0

    if rank == 0:
        log_file = Path(config.log_dir) / f"{config.model_identifier}_training.log"
    else:
        log_file = Path(config.log_dir) / f"{config.model_identifier}_training_rank{rank}.log"

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

    # Dispatch to appropriate training mode
    logger.info("Training mode: %s", args.mode)

    if args.mode == 'cv':
        # Cross-validation mode for hyperparameter tuning
        if use_mpi:
            logger.info(f"Using MPI-parallel cross-validation with {world_size} processes")
            train_cross_validation_mpi(config, dataset, args)
        else:
            if MPI_AVAILABLE and world_size == 1:
                logger.info("Only 1 MPI process detected, using sequential cross-validation")
            else:
                logger.info("MPI not available, using sequential cross-validation")
            train_cross_validation_sequential(config, dataset, args)
    elif args.mode == 'final':
        # Final training mode on all data
        train_final_model(config, dataset, args)
    else:
        logger.error("Invalid training mode: %s", args.mode)
        sys.exit(1)


if __name__ == '__main__':
    main()
