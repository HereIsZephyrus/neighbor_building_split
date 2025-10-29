"""Inference script for generating embeddings from trained GAT model.

Usage:
    python -m src.gat.inference --checkpoint models/gat/best_model.pth --output-dir output/embeddings
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pickle

import torch
import numpy as np
from tqdm import tqdm

from .models.gat import GAT
from .data.data_utils import load_district_graph
from .utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from trained GAT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='output/voronoi',
        help='Directory containing adjacency matrices'
    )
    parser.add_argument(
        '--building-shapefile',
        type=str,
        required=True,
        help='Path to building shapefile'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/gat/embeddings',
        help='Directory to save embeddings'
    )
    parser.add_argument(
        '--district-ids',
        type=int,
        nargs='+',
        default=None,
        help='List of district IDs to process (default: all found in data-dir)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--batch-inference',
        action='store_true',
        help='Use batch inference for large graphs (currently full-graph only)'
    )

    return parser.parse_args()


def auto_detect_district_ids(data_dir: Path) -> List[int]:
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


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> tuple:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    logger = get_logger()
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config_dict = checkpoint.get('config', {})
    model_config = config_dict.get('model', {})

    # Create model
    model = GAT(
        in_features=model_config.get('in_features', 12),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_classes=model_config.get('num_classes', 3),
        num_layers=model_config.get('num_layers', 3),
        num_heads=model_config.get('num_heads', 8),
        dropout=model_config.get('dropout', 0.6),
        negative_slope=model_config.get('negative_slope', 0.2),
        add_self_loops=model_config.get('add_self_loops', True)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Model:\n{model}")

    return model, config_dict


def generate_embeddings_for_district(
    model: GAT,
    district_id: int,
    data_dir: Path,
    building_shapefile: Path,
    device: str,
    scaler=None
) -> tuple:
    """
    Generate embeddings for a single district.

    Args:
        model: Trained GAT model
        district_id: District ID
        data_dir: Data directory
        building_shapefile: Path to building shapefile
        device: Device
        scaler: Optional pre-fitted StandardScaler

    Returns:
        Tuple of (embeddings, labels, num_nodes)
    """
    logger = get_logger()

    try:
        # Load district data
        data, scaler = load_district_graph(
            district_id=district_id,
            data_dir=data_dir,
            building_shapefile_path=building_shapefile,
            normalize_features=True,
            scaler=scaler
        )

        # Move to device
        data = data.to(device)

        # Generate embeddings and predict cluster count
        with torch.no_grad():
            embeddings, num_clusters_pred = model.forward_inference(data.x, data.edge_index)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        labels_np = data.y.cpu().numpy()
        num_clusters_pred_val = int(torch.round(num_clusters_pred).item())
        true_num_clusters = int(data.num_clusters.item()) if hasattr(data, 'num_clusters') else None

        logger.info(
            f"Generated embeddings for district {district_id}: "
            f"shape={embeddings_np.shape}, labels={len(np.unique(labels_np))} classes, "
            f"predicted_clusters={num_clusters_pred_val}, true_clusters={true_num_clusters}"
        )

        return embeddings_np, labels_np, data.num_nodes, num_clusters_pred_val, scaler

    except Exception as e:
        logger.error(f"Failed to generate embeddings for district {district_id}: {e}")
        return None, None, 0, None, scaler


def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    district_id: int,
    output_dir: Path,
    predicted_num_clusters: int = None
) -> None:
    """
    Save embeddings to pickle file.

    Args:
        embeddings: Node embeddings (N, D)
        labels: Node labels (N,)
        district_id: District ID
        output_dir: Output directory
        predicted_num_clusters: Predicted number of clusters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'district_{district_id}_embeddings.pkl'

    data = {
        'embeddings': embeddings,
        'labels': labels,
        'district_id': district_id,
        'num_nodes': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'predicted_num_clusters': predicted_num_clusters
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    logger = get_logger()
    logger.info(f"Embeddings saved to {output_file}")


def main():
    """Main inference function."""
    args = parse_args()

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    building_shapefile = Path(args.building_shapefile)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    if not building_shapefile.exists():
        print(f"Error: Building shapefile not found: {building_shapefile}")
        sys.exit(1)

    # Setup logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'inference.log'
    logger = setup_logger(name='gat', log_file=log_file)

    logger.info("=" * 80)
    logger.info("GAT Inference: Generating Embeddings")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {args.device}")

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, args.device)

    district_ids = get_building_district_ids(building_shapefile)

    logger.info(f"Processing {len(district_ids)} districts: {district_ids}")

    # Generate embeddings for each district
    all_embeddings = {}
    scaler = None  # Will be fitted on first district, then reused

    for district_id in tqdm(district_ids, desc="Generating embeddings"):
        embeddings, labels, num_nodes, predicted_num_clusters, scaler = generate_embeddings_for_district(
            model=model,
            district_id=district_id,
            data_dir=data_dir,
            building_shapefile=building_shapefile,
            device=args.device,
            scaler=scaler
        )

        if embeddings is not None:
            # Save embeddings
            save_embeddings(embeddings, labels, district_id, output_dir, predicted_num_clusters)

            all_embeddings[district_id] = {
                'embeddings': embeddings,
                'labels': labels,
                'num_nodes': num_nodes,
                'predicted_num_clusters': predicted_num_clusters
            }

    # Save summary
    summary = {
        'num_districts': len(all_embeddings),
        'total_nodes': sum(d['num_nodes'] for d in all_embeddings.values()),
        'embedding_dim': model.embedding_dim,
        'district_ids': list(all_embeddings.keys())
    }

    summary_file = output_dir / 'embeddings_summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)

    logger.info("=" * 80)
    logger.info("Embedding generation completed!")
    logger.info(f"Processed {summary['num_districts']} districts")
    logger.info(f"Total nodes: {summary['total_nodes']}")
    logger.info(f"Embedding dimension: {summary['embedding_dim']}")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

