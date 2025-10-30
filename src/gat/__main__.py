"""Main entry point for GAT training and inference.

Usage:
    # Training
    python -m src.gat --train --adjacency-dir /path/to/voronoi --sample-buildings /path/to/buildings.shp --sample-districts /path/to/districts.shp --output-root-dir /path/to/output

    # Inference
    python -m src.gat --inference --model-path models/final_model.pth --building-path /path/to/buildings.shp --adjacency-dir /path/to/voronoi --output-root-dir output/embeddings
"""

import argparse
import sys
import torch
from .train import main as train_main
from .inference import main as inference_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GAT model for building clustering - Training and Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Use --train and --inference as flags to determine mode
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training mode'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Run inference mode'
    )
    parser.add_argument(
        '--adjacency-dir',
        type=str,
        help='Directory containing adjacency matrices (pkl files)'
    )
    parser.add_argument(
        '--output-root-dir',
        type=str,
        help='Directory for all outputs (checkpoints, logs, embeddings)'
    )

    # Training arguments

    parser.add_argument(
        '--sample-buildings',
        type=str,
        help='Path to building shapefile [train mode]'
    )
    parser.add_argument(
        '--sample-districts',
        type=str,
        help='Path to district shapefile [train mode]'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config YAML file (default: src/gat/training_config.yaml) [train mode]'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from [train mode]'
    )

    # Inference arguments
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model checkpoint [inference mode]'
    )
    parser.add_argument(
        '--building-path',
        type=str,
        help='Path to building shapefile [inference mode]'
    )
    parser.add_argument(
        '--district-ids',
        type=int,
        nargs='+',
        default=None,
        help='List of district IDs to process (default: all found in data-dir) [inference mode]'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu, default: auto-detect) [inference mode]'
    )
    parser.add_argument(
        '--batch-inference',
        action='store_true',
        help='Use batch inference for large graphs (currently full-graph only) [inference mode]'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.train and args.inference:
        print("Error: Cannot specify both --train and --inference")
        sys.exit(1)

    if not args.train and not args.inference:
        print("Error: Must specify either --train or --inference")
        print("Use --help for more information")
        sys.exit(1)

    if args.train:
        # Validate required training arguments
        if not args.adjacency_dir:
            print("Error: --adjacency-dir is required for training")
            sys.exit(1)
        if not args.sample_buildings:
            print("Error: --sample-buildings is required for training")
            sys.exit(1)
        if not args.sample_districts:
            print("Error: --sample-districts is required for training")
            sys.exit(1)
        if not args.output_root_dir:
            print("Error: --output-root-dir is required for training")
            sys.exit(1)

        # Create a namespace with only training arguments
        train_args = argparse.Namespace(
            adjacency_dir=args.adjacency_dir,
            sample_buildings=args.sample_buildings,
            sample_districts=args.sample_districts,
            output_root_dir=args.output_root_dir,
            config=args.config,
            resume=args.resume
        )
        train_main(train_args)

    elif args.inference:
        # Validate required inference arguments
        if not args.model_path:
            print("Error: --checkpoint is required for inference")
            sys.exit(1)
        if not args.building_path:
            print("Error: --building-path is required for inference")
            sys.exit(1)

        # Create a namespace with only inference arguments
        inference_args = argparse.Namespace(
            model_path=args.model_path,
            adjacency_dir=args.adjacency_dir,
            building_path=args.building_path,
            output_root_dir=args.output_root_dir,
            district_ids=args.district_ids,
            device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
            batch_inference=args.batch_inference
        )
        inference_main(inference_args)

if __name__ == '__main__':
    main()
