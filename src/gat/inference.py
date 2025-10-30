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
import geopandas as gpd
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
        '--building-path',
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
        building_path: Path to building shapefile
        device: Device
        scaler: Optional pre-fitted StandardScaler
        perform_clustering: Whether to perform spectral clustering
        n_clusters: Number of clusters (auto-estimate if None)

    Returns:
        Tuple of (embeddings, logits, gat_labels, spectral_clusters, 
                  cluster_to_label, original_labels, num_nodes, n_clusters, scaler, has_labels)
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

        # Check if ground truth labels exist
        has_labels = getattr(data, 'has_labels', False)

        # Log generation info
        if has_labels:
            true_num_clusters = int(data.num_clusters.item())
            logger.info(
                "Generated embeddings for district %d: shape=%s, GAT predicted %d classes, ground_truth=%d classes",
                district_id, embeddings_np.shape, len(np.unique(gat_labels_np)), true_num_clusters
            )
        else:
            logger.info(
                "Generated embeddings for district %d: shape=%s, GAT predicted %d classes (no ground truth)",
                district_id, embeddings_np.shape, len(np.unique(gat_labels_np))
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
            scaler,
            has_labels
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to generate embeddings for district %d: %s", district_id, exc, exc_info=True)
        return None, None, None, None, None, None, 0, None, scaler, False


def save_embeddings(
    embeddings: np.ndarray,
    logits: np.ndarray,
    gat_labels: np.ndarray,
    original_labels: np.ndarray,
    district_id: int,
    output_dir: Path,
    spectral_clusters: Optional[np.ndarray] = None,
    cluster_to_label: Optional[dict] = None,
    predicted_num_clusters: Optional[int] = None,
    has_ground_truth: bool = False
) -> None:
    """
    Save embeddings and clustering results to pickle file.

    Args:
        embeddings: GAT node embeddings (N, D)
        logits: GAT classification logits (N, num_classes)
        gat_labels: GAT predicted labels (N,)
        original_labels: Original labels (N,) - dummy zeros if no ground truth
        district_id: District ID
        output_dir: Output directory
        spectral_clusters: Spectral cluster assignments (N,)
        cluster_to_label: Mapping from cluster ID to GAT label
        predicted_num_clusters: Number of clusters detected/predicted
        has_ground_truth: Whether ground truth labels exist
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'district_{district_id}_embeddings.pkl'

    data = {
        'embeddings': embeddings,
        'logits': logits,
        'gat_labels': gat_labels,
        'spectral_clusters': spectral_clusters,
        'cluster_to_label': cluster_to_label,
        'district_id': district_id,
        'num_nodes': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'predicted_num_clusters': predicted_num_clusters,
        'has_ground_truth': has_ground_truth
    }

    # Only save original labels if ground truth exists
    if has_ground_truth:
        data['original_labels'] = original_labels

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    logger = get_logger()
    mode_str = " (with ground truth)" if has_ground_truth else " (inference only)"
    logger.info("Embeddings and clustering results%s saved to %s", mode_str, output_file)


def update_gpkg_with_district(
    district_id: int,
    predictions: dict,
    building_path: Path,
    adjacency_dir: Path,
    output_dir: Path
) -> None:
    """
    Update GeoPackage file with predictions for a single district.
    Uses append mode to incrementally update the file.

    Args:
        district_id: District ID
        predictions: Prediction data for this district
        building_path: Path to original building shapefile
        adjacency_dir: Directory containing adjacency matrices
        output_dir: Output directory for GeoPackage
    """
    logger = get_logger()

    try:
        # Load building shapefile
        buildings_gdf = gpd.read_file(building_path)

        # Find building ID field
        id_field = None
        for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'fid', 'building_id']:
            if possible_id in buildings_gdf.columns:
                id_field = possible_id
                break

        if id_field is None:
            buildings_gdf['building_id'] = buildings_gdf.index
            id_field = 'building_id'

        # Load adjacency matrix to get building IDs for this district
        adjacency_path = adjacency_dir / f"district_{district_id}_adjacency.pkl"
        if not adjacency_path.exists():
            logger.warning("Adjacency file not found for district %d", district_id)
            return

        adjacency_matrix = pd.read_pickle(adjacency_path)
        building_ids = adjacency_matrix.index.tolist()

        # Get predictions for this district
        gat_labels = predictions['gat_labels']
        spectral_clusters = predictions.get('spectral_clusters')

        # Create mapping for this district's buildings
        building_predictions = {}
        for idx, bid in enumerate(building_ids):
            if idx < len(gat_labels):
                building_predictions[bid] = {
                    'district_id': district_id,
                    'gat_label': int(gat_labels[idx]),
                    'spectral_cluster': int(spectral_clusters[idx]) if spectral_clusters is not None else None
                }

        # Filter buildings to only those in this district
        district_building_ids = list(building_predictions.keys())
        district_buildings = buildings_gdf[buildings_gdf[id_field].isin(district_building_ids)].copy()

        if len(district_buildings) == 0:
            logger.warning("No buildings found for district %d", district_id)
            return

        # Add prediction columns
        district_buildings['district_id'] = district_id
        district_buildings['gat_label'] = district_buildings[id_field].map(
            lambda x: building_predictions.get(x, {}).get('gat_label')
        )
        district_buildings['spectral_cluster'] = district_buildings[id_field].map(
            lambda x: building_predictions.get(x, {}).get('spectral_cluster')
        )

        # Convert to appropriate types
        district_buildings['district_id'] = district_buildings['district_id'].astype('Int64')
        district_buildings['gat_label'] = district_buildings['gat_label'].astype('Int64')
        if district_buildings['spectral_cluster'].notna().any():
            district_buildings['spectral_cluster'] = district_buildings['spectral_cluster'].astype('Int64')

        # Save to GeoPackage (append mode if file exists)
        output_file = output_dir / 'building_predictions.gpkg'

        if output_file.exists():
            # Append to existing file
            existing_gdf = gpd.read_file(output_file)

            # Remove any existing records for this district (in case of re-run)
            existing_gdf = existing_gdf[existing_gdf['district_id'] != district_id]

            # Combine with new data
            combined_gdf = pd.concat([existing_gdf, district_buildings], ignore_index=True)
            combined_gdf.to_file(output_file, driver='GPKG')

            logger.info("Updated GeoPackage with district %d: %d buildings (total: %d buildings)", 
                       district_id, len(district_buildings), len(combined_gdf))
        else:
            # Create new file
            district_buildings.to_file(output_file, driver='GPKG')
            logger.info("Created GeoPackage with district %d: %d buildings", 
                       district_id, len(district_buildings))

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to update GeoPackage for district %d: %s", district_id, exc, exc_info=True)


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

    # Check if resume mode is enabled
    resume_mode = getattr(args, 'resume', None) is not None
    if resume_mode:
        logger.info("Resume mode enabled: will skip districts with existing pkl files in %s", output_dir)

    for district_id in tqdm(district_ids, desc="Generating embeddings and clustering"):
        # Check if this district has already been processed (for resume mode)
        output_pkl = output_dir / f'district_{district_id}_embeddings.pkl'
        if resume_mode and output_pkl.exists():
            logger.info("Skipping district %d (already processed, found %s)", district_id, output_pkl)

            # Load existing embeddings to include in summary
            try:
                with open(output_pkl, 'rb') as f:
                    existing_data = pickle.load(f)
                all_embeddings[district_id] = existing_data
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to load existing embeddings for district %d: %s", district_id, exc)

            continue

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
         cluster_to_label, original_labels, num_nodes, n_clusters, scaler, has_labels) = result

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
                predicted_num_clusters=n_clusters,
                has_ground_truth=has_labels
            )

            embeddings_dict = {
                'embeddings': embeddings,
                'logits': logits,
                'gat_labels': gat_labels,
                'spectral_clusters': spectral_clusters,
                'cluster_to_label': cluster_to_label,
                'num_nodes': num_nodes,
                'predicted_num_clusters': n_clusters,
                'has_ground_truth': has_labels
            }
            # Only include original labels if ground truth exists
            if has_labels:
                embeddings_dict['original_labels'] = original_labels

            all_embeddings[district_id] = embeddings_dict

            # Incrementally update GeoPackage with this district's predictions
            update_gpkg_with_district(
                district_id=district_id,
                predictions=embeddings_dict,
                building_path=building_path,
                adjacency_dir=adjacency_dir,
                output_dir=output_dir
            )

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

    # Log final GeoPackage info
    gpkg_file = output_dir / 'building_predictions.gpkg'
    if gpkg_file.exists():
        final_gdf = gpd.read_file(gpkg_file)
        logger.info("=" * 80)
        logger.info("GeoPackage Summary:")
        logger.info("  - Total buildings: %d", len(final_gdf))
        logger.info("  - Districts: %s", sorted(final_gdf['district_id'].unique().tolist()))
        logger.info("  - GAT labels: %d unique classes", final_gdf['gat_label'].nunique())
        if final_gdf['spectral_cluster'].notna().any():
            logger.info("  - Spectral clusters: %d unique clusters", final_gdf['spectral_cluster'].nunique())
        logger.info("  - File: %s", gpkg_file)

    logger.info("=" * 80)
    logger.info("Embedding generation completed!")
    logger.info("Processed %d districts", summary['num_districts'])
    logger.info("Total nodes: %d", summary['total_nodes'])
    logger.info("Embedding dimension: %d", summary['embedding_dim'])
    logger.info("Summary saved to %s", summary_file)
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
