"""TensorBoard utilities for GAT training."""

from typing import Dict, List, Optional
from pathlib import Path
import io
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import PIL.Image
import yaml

from ..utils import get_logger
from ..utils.spectral_clustering import perform_spectral_clustering_pipeline
from ..utils.feature_extractor import extract_clustering_features

matplotlib.use('Agg')  # Use non-interactive backend
logger = get_logger(__name__)

def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    epoch: int,
    prefix: str = 'train'
) -> None:
    """
    Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary of metrics to log
        epoch: Current epoch
        prefix: Prefix for metric names (e.g., 'train', 'val')
    """
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, epoch)


def visualize_district_predictions(
    district_id: int,
    buildings_gdf: gpd.GeoDataFrame,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    spectral_predictions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Visualize building predictions for a district.

    Args:
        district_id: District ID
        buildings_gdf: GeoDataFrame containing building geometries
        predictions: GAT predicted labels for each building
        ground_truth: Ground truth labels for each building
        num_classes: Number of classes
        spectral_predictions: Optional spectral clustering predictions

    Returns:
        Image as numpy array (H, W, C) in RGB format
    """
    # Create color map for labels
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    cmap = ListedColormap(colors)

    # Create figure with 2 or 3 subplots depending on whether spectral predictions exist
    n_cols = 3 if spectral_predictions is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))

    # Get plot bounds
    minx, miny, maxx, maxy = buildings_gdf.total_bounds
    buffer_pct = 0.05
    width = maxx - minx
    height = maxy - miny
    plot_bounds = [
        minx - width * buffer_pct,
        maxx + width * buffer_pct,
        miny - height * buffer_pct,
        maxy + height * buffer_pct
    ]

    # Create temporary GeoDataFrames with labels
    gdf_gt = buildings_gdf.copy()
    gdf_pred = buildings_gdf.copy()
    gdf_gt['label'] = ground_truth
    gdf_pred['label'] = predictions

    # Plot ground truth
    ax_idx = 0
    ax1 = axes[ax_idx] if n_cols > 1 else axes
    gdf_gt.plot(ax=ax1, column='label', cmap=cmap, 
                alpha=0.7, edgecolor='black', linewidth=0.5,
                legend=True, vmin=0, vmax=num_classes-1)
    ax1.set_xlim(plot_bounds[0], plot_bounds[1])
    ax1.set_ylim(plot_bounds[2], plot_bounds[3])
    ax1.set_title(f'District {district_id}: Ground Truth\n({len(buildings_gdf)} buildings)', 
                  fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Plot GAT predictions
    ax_idx += 1
    ax2 = axes[ax_idx] if n_cols > 1 else axes
    gdf_pred.plot(ax=ax2, column='label', cmap=cmap,
                  alpha=0.7, edgecolor='black', linewidth=0.5,
                  legend=True, vmin=0, vmax=num_classes-1)
    ax2.set_xlim(plot_bounds[0], plot_bounds[1])
    ax2.set_ylim(plot_bounds[2], plot_bounds[3])

    # Calculate accuracy
    gat_accuracy = (predictions == ground_truth).mean() * 100
    ax2.set_title(f'District {district_id}: GAT Direct\nAccuracy: {gat_accuracy:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Plot spectral clustering predictions if available
    if spectral_predictions is not None:
        ax_idx += 1
        ax3 = axes[ax_idx]
        gdf_spectral = buildings_gdf.copy()
        gdf_spectral['label'] = spectral_predictions
        gdf_spectral.plot(ax=ax3, column='label', cmap=cmap,
                          alpha=0.7, edgecolor='black', linewidth=0.5,
                          legend=True, vmin=0, vmax=num_classes-1)
        ax3.set_xlim(plot_bounds[0], plot_bounds[1])
        ax3.set_ylim(plot_bounds[2], plot_bounds[3])

        spectral_accuracy = (spectral_predictions == ground_truth).mean() * 100
        ax3.set_title(f'District {district_id}: Spectral Clustering\nAccuracy: {spectral_accuracy:.1f}%', 
                      fontsize=12, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.axis('off')

    plt.tight_layout()

    # Convert figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image_array = np.array(image)

    plt.close(fig)

    # Convert RGBA to RGB if needed
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    return image_array


def log_district_visualizations_to_tensorboard(
    writer: SummaryWriter,
    model: torch.nn.Module,
    data_list: List[Data],
    building_path: Path,
    epoch: int,
    tag: str = 'train',
    max_districts: int = 5,
    device: str = 'cuda',
    district_path: Path = None,
    adjacency_dir: Optional[Path] = None,
    enable_spectral_clustering: bool = True
) -> None:
    """
    Log district visualizations to TensorBoard.

    Follows inference.py pipeline:
    1. GAT forward pass to get embeddings and logits
    2. Spectral clustering on embeddings + features
    3. Confidence-weighted voting to map clusters to labels
    4. Visualize: Ground Truth vs GAT Direct vs Spectral Clustering

    Args:
        writer: TensorBoard SummaryWriter
        model: GAT model
        data_list: List of PyG Data objects (districts)
        building_path: Path to building shapefile
        epoch: Current epoch number
        tag: Tag prefix for TensorBoard (e.g., 'train', 'val')
        max_districts: Maximum number of districts to visualize
        device: Device to run model on
        district_path: Path to district shapefile (for spatial matching)
        adjacency_dir: Directory containing adjacency matrices (required for spectral clustering)
        enable_spectral_clustering: Whether to perform spectral clustering (先聚类再分类)
    """
    if not data_list:
        logger.warning("No data provided for visualization")
        return

    logger.info(f"Generating {tag} visualizations for up to {max_districts} districts...")

    # Load building geometries
    try:
        buildings_all = gpd.read_file(building_path)
    except Exception as e:
        logger.error(f"Failed to load building geometries: {e}")
        return

    # Load district geometries if provided
    districts_gdf = None
    if district_path and district_path.exists():
        try:
            districts_gdf = gpd.read_file(district_path)
            logger.info(f"Loaded district geometries from {district_path}")
        except Exception as e:
            logger.warning(f"Failed to load district geometries: {e}. Will try field-based matching.")

    # Load spectral clustering config
    spectral_config = {}
    if enable_spectral_clustering:
        try:
            parent_config_path = Path(__file__).parent.parent / 'training_config.yaml'
            if parent_config_path.exists():
                with open(parent_config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                spectral_config = full_config.get('spectral_clustering', {})
                logger.debug("Loaded spectral clustering config from training_config.yaml")
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to load spectral clustering config: %s", e)

        # Set default values if not loaded
        spectral_config.setdefault('embedding_weight', 0.3)
        spectral_config.setdefault('feature_weight', 0.5)
        spectral_config.setdefault('distance_weight', 0.2)
        spectral_config.setdefault('distance_scale', 100.0)
        spectral_config.setdefault('use_confidence_weighted_voting', True)
        spectral_config.setdefault('area_threshold_m2', 1_000_000)

    # Set model to evaluation mode
    model.eval()
    device_obj = torch.device(device)

    # Visualize up to max_districts
    num_visualize = min(len(data_list), max_districts)

    with torch.no_grad():
        for i, data in enumerate(data_list[:num_visualize]):
            try:
                # Get district info
                district_id = data.district_id if hasattr(data, 'district_id') else i

                # Get GAT predictions and embeddings
                data_device = data.to(device_obj)

                # Check if model has forward_inference method (for embeddings)
                if hasattr(model, 'forward_inference'):
                    logits, embeddings = model.forward_inference(data_device.x, data_device.edge_index)
                else:
                    logits = model(data_device.x, data_device.edge_index)
                    embeddings = None  # No embeddings available

                gat_predictions = logits.argmax(dim=1).cpu().numpy()
                ground_truth = data.y.cpu().numpy()

                # Convert to numpy for spectral clustering
                logits_np = logits.cpu().numpy()
                embeddings_np = embeddings.cpu().numpy() if embeddings is not None else None

                # Get building geometries for this district
                district_buildings = None

                # Method 1: Use spatial matching with district geometry (preferred)
                if districts_gdf is not None:
                    try:
                        district_geom = districts_gdf[districts_gdf['FID'] == district_id].geometry
                        if len(district_geom) > 0:
                            district_geom = district_geom.iloc[0]
                            # Use spatial intersection to get buildings
                            district_buildings = buildings_all[buildings_all.intersects(district_geom)].copy()
                            logger.debug(f"District {district_id}: Found {len(district_buildings)} buildings using spatial matching")
                    except Exception as e:
                        logger.debug(f"Spatial matching failed for district {district_id}: {e}")

                # Method 2: Fall back to field-based matching
                if district_buildings is None or len(district_buildings) == 0:
                    district_id_field = None
                    for field in ['district_id', 'FID', 'id', 'TAZ_ID']:
                        if field in buildings_all.columns:
                            district_id_field = field
                            break

                    if district_id_field is not None:
                        district_buildings = buildings_all[buildings_all[district_id_field] == district_id].copy()
                        logger.debug(f"District {district_id}: Found {len(district_buildings)} buildings using field '{district_id_field}'")

                if district_buildings is None or len(district_buildings) == 0:
                    logger.warning(f"No buildings found for district {district_id}")
                    continue

                # Ensure we have the same number of buildings
                if len(district_buildings) != len(gat_predictions):
                    logger.warning(
                        f"Mismatch in building count for district {district_id}: "
                        f"shapefile={len(district_buildings)}, predictions={len(gat_predictions)}"
                    )
                    # Try to match by taking first N buildings
                    min_len = min(len(district_buildings), len(gat_predictions))
                    district_buildings = district_buildings.iloc[:min_len]
                    gat_predictions = gat_predictions[:min_len]
                    ground_truth = ground_truth[:min_len]
                    if embeddings_np is not None:
                        embeddings_np = embeddings_np[:min_len]
                    logits_np = logits_np[:min_len]

                # Perform spectral clustering (先聚类再分类)
                spectral_predictions = None
                if enable_spectral_clustering and embeddings_np is not None and adjacency_dir is not None:
                    try:
                        # Load adjacency matrix
                        adjacency_path = adjacency_dir / f"district_{district_id}_adjacency.pkl"
                        if adjacency_path.exists():
                            adjacency_matrix = pd.read_pickle(adjacency_path)
                            building_ids = adjacency_matrix.index.tolist()

                            # Extract clustering features
                            clustering_features = extract_clustering_features(district_buildings)

                            # Load voronoi areas if available
                            voronoi_areas = None
                            if 'voroniarea' in district_buildings.columns:
                                voronoi_areas = dict(zip(building_ids, district_buildings['voroniarea'].values))

                            # Perform spectral clustering
                            spectral_clusters, _, cluster_to_label, _ = perform_spectral_clustering_pipeline(
                                embeddings=embeddings_np,
                                features=clustering_features,
                                adjacency_matrix=adjacency_matrix,
                                gat_labels=gat_predictions,
                                gat_logits=logits_np,
                                building_ids=building_ids,
                                voronoi_areas=voronoi_areas,
                                n_clusters=None,  # Auto-detect
                                use_confidence_weighted_voting=spectral_config['use_confidence_weighted_voting'],
                                embedding_weight=spectral_config['embedding_weight'],
                                feature_weight=spectral_config['feature_weight'],
                                distance_weight=spectral_config['distance_weight'],
                                distance_scale=spectral_config['distance_scale'],
                                area_threshold_m2=spectral_config['area_threshold_m2'],
                                random_state=42
                            )

                            # Map clusters to labels
                            spectral_predictions = np.array([cluster_to_label[c] for c in spectral_clusters])
                            logger.debug(f"District {district_id}: Spectral clustering completed, "
                                       f"{len(np.unique(spectral_clusters))} clusters -> {len(np.unique(spectral_predictions))} labels")
                        else:
                            logger.debug(f"Adjacency matrix not found: {adjacency_path}")
                    except Exception as e:
                        logger.warning(f"Spectral clustering failed for district {district_id}: {e}")
                        spectral_predictions = None

                # Get number of classes from data
                num_classes = int(ground_truth.max()) + 1

                # Generate visualization
                image_array = visualize_district_predictions(
                    district_id=district_id,
                    buildings_gdf=district_buildings,
                    predictions=gat_predictions,
                    ground_truth=ground_truth,
                    num_classes=num_classes,
                    spectral_predictions=spectral_predictions
                )

                # Add to TensorBoard (HWC -> CHW format)
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                writer.add_image(
                    f'{tag}_predictions/district_{district_id}',
                    image_tensor,
                    epoch
                )

                logger.info(
                    f"Added visualization for district {district_id} "
                    f"({len(district_buildings)} buildings)"
                )

            except Exception as e:
                logger.error(f"Failed to visualize district {i}: {e}", exc_info=True)
                continue

    logger.info(f"Completed {tag} visualizations")
