"""Inference script for generating embeddings from trained GAT model.

Usage:
    python -m src.gat.inference --checkpoint models/gat/best_model.pth --output-root-dir output/embeddings
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pickle

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models.gat import GAT
from .data.data_utils import load_district_graph
from .utils.logger import setup_logger, get_logger
from .utils.spectral_clustering import perform_spectral_clustering_pipeline


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
        '--adjacency-dir',
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
        '--output-root-dir',
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


def load_model_from_file(checkpoint_path: Path, device: str) -> tuple:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    logger = get_logger()
    logger.info("Loading checkpoint from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config_dict = checkpoint.get('config', {})
    model_config = config_dict.get('model', {})

    # Create model
    model = GAT(
        in_features=model_config.get('in_features', 13),
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

    logger.info("Model loaded from epoch %s", checkpoint.get('epoch', 'unknown'))
    logger.info("Model:\n%s", model)

    return model, config_dict


def generate_embeddings_for_district(
    model: GAT,
    district_id: int,
    adjacency_dir: Path,
    building_path: Path,
    device: str,
    scaler=None,
    perform_clustering: bool = True,
    n_clusters: Optional[int] = None
) -> tuple:
    """
    Generate embeddings for a single district and optionally perform spectral clustering.

    Args:
        model: Trained GAT model
        district_id: District ID
        adjacency_dir: Data directory
        building_shapefile: Path to building shapefile
        device: Device
        scaler: Optional pre-fitted StandardScaler
        perform_clustering: Whether to perform spectral clustering
        n_clusters: Number of clusters (auto-estimate if None)

    Returns:
        Tuple of (embeddings, logits, gat_labels, spectral_clusters, 
                  cluster_to_label, original_labels, num_nodes, n_clusters, scaler)
    """
    logger = get_logger()

    try:
        # Load district data (normalized features)
        data, scaler = load_district_graph(
            district_id=district_id,
            adjacency_dir=adjacency_dir,
            building_path=building_path,
            normalize_features=True,
            scaler=scaler
        )

        # Also load unnormalized data to get original features
        data_unnorm, _ = load_district_graph(
            district_id=district_id,
            adjacency_dir=adjacency_dir,
            building_path=building_path,
            normalize_features=False,
            scaler=None
        )
        original_features = data_unnorm.x.numpy()

        # Load adjacency matrix
        adjacency_path = adjacency_dir / f"district_{district_id}_adjacency.pkl"
        adjacency_matrix = pd.read_pickle(adjacency_path)

        # Move to device
        data = data.to(device)

        # GAT forward pass: get logits and embeddings
        with torch.no_grad():
            logits, embeddings = model.forward_inference(data.x, data.edge_index)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        logits_np = logits.cpu().numpy()
        original_labels_np = data.y.cpu().numpy()

        # Get GAT predicted labels
        gat_labels_np = torch.argmax(logits, dim=1).cpu().numpy()

        true_num_clusters = int(data.num_clusters.item()) if hasattr(data, 'num_clusters') else None

        logger.info(
            "Generated embeddings for district %d: shape=%s, GAT predicted %d classes, true_clusters=%s",
            district_id, embeddings_np.shape, len(np.unique(gat_labels_np)), true_num_clusters
        )

        # Perform spectral clustering
        spectral_clusters = None
        cluster_to_label = None
        final_n_clusters = n_clusters

        if perform_clustering:
            try:
                spectral_clusters, _, cluster_to_label, _ = perform_spectral_clustering_pipeline(
                    embeddings=embeddings_np,
                    features=original_features,
                    adjacency_matrix=adjacency_matrix,
                    gat_labels=gat_labels_np,
                    n_clusters=n_clusters,
                    embedding_weight=0.5,
                    feature_weight=0.3,
                    distance_weight=0.2,
                    distance_scale=100.0,  # Adjust based on distance units
                    random_state=42
                )

                final_n_clusters = len(np.unique(spectral_clusters))

                logger.info(
                    "Spectral clustering completed for district %d: %d clusters, cluster_to_label=%s",
                    district_id, final_n_clusters, cluster_to_label
                )

            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Spectral clustering failed for district %d: %s", district_id, exc, exc_info=True)
                spectral_clusters = None
                cluster_to_label = None

        return (
            embeddings_np, 
            logits_np, 
            gat_labels_np, 
            spectral_clusters, 
            cluster_to_label,
            original_labels_np, 
            data.num_nodes, 
            final_n_clusters,
            scaler
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to generate embeddings for district %d: %s", district_id, exc, exc_info=True)
        return None, None, None, None, None, None, 0, None, scaler


def save_embeddings(
    embeddings: np.ndarray,
    logits: np.ndarray,
    gat_labels: np.ndarray,
    original_labels: np.ndarray,
    district_id: int,
    output_dir: Path,
    spectral_clusters: Optional[np.ndarray] = None,
    cluster_to_label: Optional[dict] = None,
    predicted_num_clusters: Optional[int] = None
) -> None:
    """
    Save embeddings and clustering results to pickle file.

    Args:
        embeddings: GAT node embeddings (N, D)
        logits: GAT classification logits (N, num_classes)
        gat_labels: GAT predicted labels (N,)
        original_labels: Original ground truth labels (N,)
        district_id: District ID
        output_dir: Output directory
        spectral_clusters: Spectral cluster assignments (N,)
        cluster_to_label: Mapping from cluster ID to GAT label
        predicted_num_clusters: Number of clusters detected/predicted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'district_{district_id}_embeddings.pkl'

    data = {
        'embeddings': embeddings,
        'logits': logits,
        'gat_labels': gat_labels,
        'original_labels': original_labels,
        'spectral_clusters': spectral_clusters,
        'cluster_to_label': cluster_to_label,
        'district_id': district_id,
        'num_nodes': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'predicted_num_clusters': predicted_num_clusters
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    logger = get_logger()
    logger.info("Embeddings and clustering results saved to %s", output_file)


def main(args=None):
    """Main inference function.
 
    Args:
        args: Optional argparse.Namespace. If None, will parse from sys.argv.
    """
    if args is None:
        args = parse_args()

    # Setup paths
    model_path = Path(args.model_path)
    adjacency_dir = Path(args.adjacency_dir)
    building_path = Path(args.building_path)
    output_dir = Path(args.output_root_dir) / "predicted"

    # Validate paths
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    if not adjacency_dir.exists():
        print(f"Error: Data directory not found: {adjacency_dir}")
        sys.exit(1)

    if not building_path.exists():
        print(f"Error: Building path not found: {building_path}")
        sys.exit(1)

    # Setup logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'inference.log'
    logger = setup_logger(name='gat', log_file=log_file)

    logger.info("=" * 80)
    logger.info("GAT Inference: Generating Embeddings")
    logger.info("=" * 80)
    logger.info("Model path: %s", model_path)
    logger.info("Adjacency directory: %s", adjacency_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", args.device)

    # Load model
    model, _ = load_model_from_file(model_path, args.device)

    # Get district IDs
    if args.district_ids:
        district_ids = args.district_ids
    else:
        district_ids = auto_detect_district_ids(adjacency_dir)

    logger.info("Processing %d districts.", len(district_ids))

    # Generate embeddings for each district
    all_embeddings = {}
    scaler = None  # Will be fitted on first district, then reused

    for district_id in tqdm(district_ids, desc="Generating embeddings and clustering"):
        result = generate_embeddings_for_district(
            model=model,
            district_id=district_id,
            adjacency_dir=adjacency_dir,
            building_path=building_path,
            device=args.device,
            scaler=scaler,
            perform_clustering=True,
            n_clusters=None  # Auto-detect optimal number
        )

        (embeddings, logits, gat_labels, spectral_clusters, 
         cluster_to_label, original_labels, num_nodes, n_clusters, scaler) = result

        if embeddings is not None:
            # Save embeddings and clustering results
            save_embeddings(
                embeddings=embeddings,
                logits=logits,
                gat_labels=gat_labels,
                original_labels=original_labels,
                district_id=district_id,
                output_dir=output_dir,
                spectral_clusters=spectral_clusters,
                cluster_to_label=cluster_to_label,
                predicted_num_clusters=n_clusters
            )

            all_embeddings[district_id] = {
                'embeddings': embeddings,
                'logits': logits,
                'gat_labels': gat_labels,
                'original_labels': original_labels,
                'spectral_clusters': spectral_clusters,
                'cluster_to_label': cluster_to_label,
                'num_nodes': num_nodes,
                'predicted_num_clusters': n_clusters
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
    logger.info("Processed %d districts", summary['num_districts'])
    logger.info("Total nodes: %d", summary['total_nodes'])
    logger.info("Embedding dimension: %d", summary['embedding_dim'])
    logger.info("Summary saved to %s", summary_file)
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

