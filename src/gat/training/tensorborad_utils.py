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
from matplotlib.patches import Polygon
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import PIL.Image
import yaml
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from ..utils import get_logger
from ..utils.spectral_clustering import perform_spectral_clustering_pipeline
from ..utils.feature_extractor import extract_clustering_features
from ..utils.graph_utils import get_connected_components
from ..utils.graph_utils_ext import (
    extract_subgraph,
    extract_subgraph_from_adjacency,
    merge_component_results,
    get_component_statistics
)

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
    spectral_predictions: Optional[np.ndarray] = None,
    spectral_clusters: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Visualize building predictions for a district.

    Args:
        district_id: District ID
        buildings_gdf: GeoDataFrame containing building geometries
        predictions: GAT predicted labels for each building
        ground_truth: Ground truth labels for each building
        num_classes: Number of classes
        spectral_predictions: Optional spectral clustering predictions (final labels)
        spectral_clusters: Optional spectral clustering cluster IDs (for drawing convex hulls)

    Returns:
        Image as numpy array (H, W, C) in RGB format
    """
    # Create color map for labels
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    cmap = ListedColormap(colors)

    # Create figure with 2 or 3 subplots depending on whether spectral predictions exist
    n_cols = 3 if spectral_predictions is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

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

        # Add cluster ID if available (for convex hull drawing)
        if spectral_clusters is not None:
            gdf_spectral['cluster'] = spectral_clusters

        # Plot buildings
        gdf_spectral.plot(ax=ax3, column='label', cmap=cmap,
                          alpha=0.7, edgecolor='black', linewidth=0.5,
                          legend=True, vmin=0, vmax=num_classes-1)

        # Draw convex hulls for each cluster
        if spectral_clusters is not None:
            unique_clusters = np.unique(spectral_clusters)
            # Use a colorblind-friendly palette for cluster boundaries
            cluster_colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters)))

            for cluster_id, color in zip(unique_clusters, cluster_colors):
                # Get all buildings in this cluster
                cluster_mask = gdf_spectral['cluster'] == cluster_id
                cluster_buildings = gdf_spectral[cluster_mask]

                if len(cluster_buildings) < 3:
                    # Need at least 3 points for a convex hull, skip if too few
                    continue

                try:
                    # Get all building centroids for this cluster
                    points = [geom.centroid for geom in cluster_buildings.geometry]

                    # Create MultiPoint and compute convex hull
                    multi_point = MultiPoint(points)
                    convex_hull = multi_point.convex_hull

                    # Extract coordinates for plotting
                    if convex_hull.geom_type == 'Polygon':
                        x, y = convex_hull.exterior.xy
                        ax3.plot(x, y, color=color, linewidth=2.5, alpha=0.8, 
                                linestyle='-', zorder=100)
                        # Add a subtle fill
                        ax3.fill(x, y, color=color, alpha=0.1, zorder=50)
                    elif convex_hull.geom_type == 'LineString':
                        # If only 2 points, it's a line
                        x, y = convex_hull.xy
                        ax3.plot(x, y, color=color, linewidth=2.5, alpha=0.8,
                                linestyle='-', zorder=100)

                except Exception as e:
                    logger.debug(f"Failed to compute convex hull for cluster {cluster_id}: {e}")
                    continue

        ax3.set_xlim(plot_bounds[0], plot_bounds[1])
        ax3.set_ylim(plot_bounds[2], plot_bounds[3])

        spectral_accuracy = (spectral_predictions == ground_truth).mean() * 100
        num_clusters = len(np.unique(spectral_clusters)) if spectral_clusters is not None else 0
        title = f'District {district_id}: Spectral Clustering\nAccuracy: {spectral_accuracy:.1f}%'
        if num_clusters > 0:
            title += f' ({num_clusters} clusters)'
        ax3.set_title(title, fontsize=12, fontweight='bold')
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

    Follows inference.py pipeline with connected component separation:
    1. GAT forward pass to get embeddings and logits
    2. Identify connected components in the graph
    3. Process each component independently:
       - Apply spectral clustering to components with >= min_component_size buildings
       - Use majority voting for smaller components
    4. Confidence-weighted voting to map clusters to labels
    5. Visualize: Ground Truth vs GAT Direct vs Spectral Clustering

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
        enable_spectral_clustering: Whether to perform spectral clustering
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
        spectral_config.setdefault('min_component_size', 3)

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

                # Extract edge_attr if available and ensure it's on the correct device
                edge_attr = None
                if hasattr(data_device, 'edge_attr') and data_device.edge_attr is not None:
                    edge_attr = data_device.edge_attr.to(device_obj)

                # Check if model has forward_inference method (for embeddings)
                if hasattr(model, 'forward_inference'):
                    logits, embeddings = model.forward_inference(data_device.x, data_device.edge_index, edge_attr)
                else:
                    logits = model(data_device.x, data_device.edge_index, edge_attr)
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

                # Perform spectral clustering with connected component separation
                spectral_predictions = None
                spectral_clusters_array = None
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

                            # Use connected component separation to ensure spatial contiguity
                            logger.debug(f"District {district_id}: Using connected component separation")

                            # Identify connected components
                            component_labels, num_components = get_connected_components(
                                data_device.edge_index,
                                data_device.num_nodes
                            )
                            component_labels_np = component_labels.cpu().numpy()
                            logger.debug(f"  Found {num_components} connected components")

                            # Process each component independently
                            component_results = []
                            for comp_id in range(num_components):
                                comp_mask = (component_labels_np == comp_id)
                                comp_size = comp_mask.sum()

                                # Extract component data
                                comp_data, _ = extract_subgraph(data_device, comp_mask, building_ids)

                                # Extract edge_attr if available and ensure it's on the correct device
                                comp_edge_attr = None
                                if hasattr(comp_data, 'edge_attr') and comp_data.edge_attr is not None:
                                    comp_edge_attr = comp_data.edge_attr.to(device_obj)

                                # GAT forward for component
                                if hasattr(model, 'forward_inference'):
                                    comp_logits, comp_embeddings = model.forward_inference(
                                        comp_data.x, comp_data.edge_index, comp_edge_attr
                                    )
                                else:
                                    comp_logits = model(comp_data.x, comp_data.edge_index, comp_edge_attr)
                                    comp_embeddings = None

                                comp_embeddings_np = comp_embeddings.cpu().numpy() if comp_embeddings is not None else None
                                comp_logits_np = comp_logits.cpu().numpy()
                                comp_gat_labels = comp_logits.argmax(dim=1).cpu().numpy()

                                # Apply spectral clustering if component is large enough
                                if comp_size >= spectral_config['min_component_size'] and comp_embeddings_np is not None:
                                    comp_building_ids = [building_ids[i] for i in range(len(comp_mask)) if comp_mask[i]]
                                    comp_clustering_features = clustering_features[comp_mask]
                                    comp_adjacency = extract_subgraph_from_adjacency(adjacency_matrix, comp_building_ids)
                                    comp_voronoi_areas = {bid: voronoi_areas[bid] for bid in comp_building_ids} if voronoi_areas else None

                                    comp_clusters, comp_final_labels, _, _, _ = perform_spectral_clustering_pipeline(
                                        embeddings=comp_embeddings_np,
                                        features=comp_clustering_features,
                                        adjacency_matrix=comp_adjacency,
                                        gat_labels=comp_gat_labels,
                                        gat_logits=comp_logits_np,
                                        building_ids=comp_building_ids,
                                        voronoi_areas=comp_voronoi_areas,
                                        n_clusters=None,
                                        use_confidence_weighted_voting=spectral_config['use_confidence_weighted_voting'],
                                        embedding_weight=spectral_config['embedding_weight'],
                                        feature_weight=spectral_config['feature_weight'],
                                        distance_weight=spectral_config['distance_weight'],
                                        distance_scale=spectral_config['distance_scale'],
                                        random_state=42
                                    )
                                else:
                                    # Small component: majority voting
                                    comp_clusters = np.zeros(comp_size, dtype=int)
                                    majority_label = np.argmax(np.bincount(comp_gat_labels))
                                    comp_final_labels = np.full(comp_size, majority_label, dtype=int)

                                component_results.append({
                                    'component_id': comp_id,
                                    'num_nodes': comp_size,
                                    'embeddings': comp_embeddings_np if comp_embeddings_np is not None else np.zeros((comp_size, embeddings_np.shape[1])),
                                    'logits': comp_logits_np,
                                    'gat_labels': comp_gat_labels,
                                    'cluster_assignments': comp_clusters,
                                    'final_labels': comp_final_labels,
                                    'node_mask': comp_mask
                                })

                            # Merge results
                            merged = merge_component_results(component_results, component_labels_np)
                            spectral_clusters_array = merged['cluster_assignments']
                            spectral_predictions = merged['final_labels']

                            logger.debug(f"District {district_id}: Connected component separation completed, "
                                       f"{num_components} components → {len(np.unique(spectral_clusters_array))} clusters")
                        else:
                            logger.debug(f"Adjacency matrix not found: {adjacency_path}")
                    except Exception as e:
                        logger.warning(f"Spectral clustering failed for district {district_id}: {e}")
                        spectral_predictions = None
                        spectral_clusters_array = None

                # Get number of classes from data
                num_classes = int(ground_truth.max()) + 1

                # Generate visualization
                image_array = visualize_district_predictions(
                    district_id=district_id,
                    buildings_gdf=district_buildings,
                    predictions=gat_predictions,
                    ground_truth=ground_truth,
                    num_classes=num_classes,
                    spectral_predictions=spectral_predictions,
                    spectral_clusters=spectral_clusters_array
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
