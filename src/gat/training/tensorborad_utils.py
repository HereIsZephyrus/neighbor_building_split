"""TensorBoard utilities for GAT training."""

from typing import Dict, List
from pathlib import Path
import io
import numpy as np
import torch
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
import PIL.Image

from ..utils import get_logger

matplotlib.use('Agg')  # Use non-interactive backend
logger = get_logger()

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
    num_classes: int
) -> np.ndarray:
    """
    Visualize building predictions for a district.

    Args:
        district_id: District ID
        buildings_gdf: GeoDataFrame containing building geometries
        predictions: Predicted labels for each building
        ground_truth: Ground truth labels for each building
        num_classes: Number of classes

    Returns:
        Image as numpy array (H, W, C) in RGB format
    """
    # Create color map for labels
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    cmap = ListedColormap(colors)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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
    ax1 = axes[0]
    gdf_gt.plot(ax=ax1, column='label', cmap=cmap, 
                alpha=0.7, edgecolor='black', linewidth=0.5,
                legend=True, vmin=0, vmax=num_classes-1)
    ax1.set_xlim(plot_bounds[0], plot_bounds[1])
    ax1.set_ylim(plot_bounds[2], plot_bounds[3])
    ax1.set_title(f'District {district_id}: Ground Truth\n({len(buildings_gdf)} buildings)', 
                  fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Plot predictions
    ax2 = axes[1]
    gdf_pred.plot(ax=ax2, column='label', cmap=cmap,
                  alpha=0.7, edgecolor='black', linewidth=0.5,
                  legend=True, vmin=0, vmax=num_classes-1)
    ax2.set_xlim(plot_bounds[0], plot_bounds[1])
    ax2.set_ylim(plot_bounds[2], plot_bounds[3])

    # Calculate accuracy
    accuracy = (predictions == ground_truth).mean() * 100
    ax2.set_title(f'District {district_id}: Predictions\nAccuracy: {accuracy:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')

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
    device: str = 'cuda'
) -> None:
    """
    Log district visualizations to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        model: GAT model
        data_list: List of PyG Data objects (districts)
        building_path: Path to building shapefile
        epoch: Current epoch number
        tag: Tag prefix for TensorBoard (e.g., 'train', 'val')
        max_districts: Maximum number of districts to visualize
        device: Device to run model on
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

                # Get predictions
                data_device = data.to(device_obj)
                logits = model(data_device.x, data_device.edge_index)
                predictions = logits.argmax(dim=1).cpu().numpy()
                ground_truth = data.y.cpu().numpy()

                # Get building geometries for this district
                # Assume buildings have a 'district_id' or 'FID' field
                district_id_field = None
                for field in ['district_id', 'FID', 'id']:
                    if field in buildings_all.columns:
                        district_id_field = field
                        break

                if district_id_field is None:
                    logger.warning(f"No district_id field found in building shapefile")
                    continue

                district_buildings = buildings_all[buildings_all[district_id_field] == district_id].copy()

                if len(district_buildings) == 0:
                    logger.warning(f"No buildings found for district {district_id}")
                    continue

                # Ensure we have the same number of buildings
                if len(district_buildings) != len(predictions):
                    logger.warning(
                        f"Mismatch in building count for district {district_id}: "
                        f"shapefile={len(district_buildings)}, predictions={len(predictions)}"
                    )
                    # Try to match by taking first N buildings
                    min_len = min(len(district_buildings), len(predictions))
                    district_buildings = district_buildings.iloc[:min_len]
                    predictions = predictions[:min_len]
                    ground_truth = ground_truth[:min_len]

                # Get number of classes from data
                num_classes = int(ground_truth.max()) + 1

                # Generate visualization
                image_array = visualize_district_predictions(
                    district_id=district_id,
                    buildings_gdf=district_buildings,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    num_classes=num_classes
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
