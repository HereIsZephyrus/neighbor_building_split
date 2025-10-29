"""Data loading and preprocessing utilities."""

import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from ..utils.feature_extractor import extract_building_features
from ..utils.graph_utils import similarity_matrix_to_edge_index
from ..utils.logger import get_logger

logger = get_logger()


def load_district_graph(
    district_id: int,
    data_dir: Path,
    building_shapefile_path: Path,
    normalize_features: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[Data, StandardScaler]:
    """
    Load district graph data from similarity matrix and shapefile.

    Args:
        district_id: District ID
        data_dir: Directory containing voronoi output (adjacency matrices)
        building_shapefile_path: Path to building shapefile
        normalize_features: Whether to normalize features
        scaler: Optional pre-fitted StandardScaler

    Returns:
        data: PyG Data object with node features, edges, and labels
        scaler: StandardScaler (fitted if not provided)
    """
    logger.debug(f"Loading district {district_id}...")

    # Load similarity/adjacency matrix
    sim_matrix_path = data_dir / f"district_{district_id}_adjacency.pkl"

    if not sim_matrix_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")

    sim_matrix = pd.read_pickle(sim_matrix_path)
    logger.debug(f"Loaded similarity matrix: shape={sim_matrix.shape}")

    # Load building shapefile
    buildings_gdf = gpd.read_file(building_shapefile_path)

    # Filter buildings to match those in similarity matrix
    building_ids_in_matrix = sim_matrix.index.tolist()

    # Create mapping from building ID to row index
    # Assumes buildings_gdf has an ID field (FID, OBJECTID, etc.)
    id_field = None
    for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'fid', 'building_id']:
        if possible_id in buildings_gdf.columns:
            id_field = possible_id
            break

    if id_field is None:
        logger.warning("No ID field found in buildings shapefile, using index")
        buildings_gdf['building_id'] = buildings_gdf.index
        id_field = 'building_id'

    # Filter to buildings in the matrix
    buildings_gdf = buildings_gdf[buildings_gdf[id_field].isin(building_ids_in_matrix)].copy()

    # Sort to match matrix order
    buildings_gdf['_sort_key'] = buildings_gdf[id_field].map({bid: i for i, bid in enumerate(building_ids_in_matrix)})
    buildings_gdf = buildings_gdf.sort_values('_sort_key').reset_index(drop=True)

    logger.debug(f"Filtered buildings: {len(buildings_gdf)} buildings")

    # Extract features
    features = extract_building_features(buildings_gdf, normalize_spatial=True)

    # Normalize features
    if normalize_features:
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            logger.debug("Fitted new StandardScaler on features")
        else:
            features = scaler.transform(features)
            logger.debug("Applied existing StandardScaler to features")

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float)

    # Convert similarity matrix to edge_index
    edge_index, edge_attr = similarity_matrix_to_edge_index(sim_matrix, threshold=None)

    # Extract labels if available
    label_field = None
    for possible_label in ['label', 'class', 'category', 'cluster', 'type']:
        if possible_label in buildings_gdf.columns:
            label_field = possible_label
            break

    if label_field is not None:
        labels = buildings_gdf[label_field].values
        # Convert to integer labels
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = torch.tensor([label_map[label] for label in labels], dtype=torch.long)
        num_clusters = len(unique_labels)
        logger.debug(f"Found labels: {num_clusters} classes")
    else:
        # No labels available
        y = torch.zeros(len(buildings_gdf), dtype=torch.long)
        num_clusters = 1
        logger.warning(f"No label field found in district {district_id}")

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(buildings_gdf),
        district_id=district_id,
        num_clusters=torch.tensor([num_clusters], dtype=torch.float)
    )

    logger.info(f"Loaded district {district_id}: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

    return data, scaler


def split_dataset(
    data_list: List[Data],
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[Data], List[Data]]:
    """
    Split dataset into train and validation sets.

    Args:
        data_list: List of PyG Data objects
        train_ratio: Ratio of training data
        random_seed: Random seed for reproducibility

    Returns:
        train_data: List of training Data objects
        val_data: List of validation Data objects
    """
    np.random.seed(random_seed)

    n_total = len(data_list)
    n_train = int(n_total * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]

    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val")

    return train_data, val_data


def compute_feature_stats(data_list: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature statistics (mean and std) across all graphs.

    Args:
        data_list: List of PyG Data objects

    Returns:
        mean: Feature means (D,)
        std: Feature standard deviations (D,)
    """
    # Collect all features
    all_features = []
    for data in data_list:
        all_features.append(data.x.numpy())

    all_features = np.concatenate(all_features, axis=0)

    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    logger.debug(f"Computed feature stats: mean={mean}, std={std}")

    return mean, std


def create_train_val_masks(
    num_nodes: int,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create train/val masks for nodes in a single graph.

    Args:
        num_nodes: Number of nodes
        train_ratio: Ratio of training nodes
        random_seed: Random seed

    Returns:
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
    """
    np.random.seed(random_seed)

    n_train = int(num_nodes * train_ratio)

    indices = np.random.permutation(num_nodes)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True

    return train_mask, val_mask


def load_multiple_districts(
    district_ids: List[int],
    data_dir: Path,
    building_shapefile_path: Path,
    normalize_features: bool = True
) -> Tuple[List[Data], StandardScaler]:
    """
    Load multiple districts and fit a common scaler.

    Args:
        district_ids: List of district IDs
        data_dir: Directory containing voronoi output
        building_shapefile_path: Path to building shapefile
        normalize_features: Whether to normalize features

    Returns:
        data_list: List of PyG Data objects
        scaler: Fitted StandardScaler
    """
    logger.info(f"Loading {len(district_ids)} districts...")

    data_list = []
    all_features = []

    # First pass: collect all features to fit scaler
    for district_id in district_ids:
        try:
            data, _ = load_district_graph(
                district_id, data_dir, building_shapefile_path,
                normalize_features=False
            )
            data_list.append(data)
            all_features.append(data.x.numpy())
        except Exception as e:
            logger.error(f"Failed to load district {district_id}: {e}")
            continue

    # Fit scaler on all features
    scaler = None
    if normalize_features and len(all_features) > 0:
        all_features_concat = np.concatenate(all_features, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_features_concat)
        logger.info("Fitted StandardScaler on all district features")

        # Normalize features
        for data in data_list:
            data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)

    logger.info(f"Loaded {len(data_list)} districts successfully")

    return data_list, scaler

