"""Data loading and preprocessing utilities."""

import pandas as pd
import geopandas as gpd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Iterator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch_geometric.data import Data

from ..utils.feature_extractor import extract_gat_features
from ..utils.graph_utils import similarity_matrix_to_edge_index
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_district_graph(
    district_id: int,
    adjacency_dir: Path,
    building_path: Path,
    normalize_features: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[Data, StandardScaler]:
    """
    Load district graph data from similarity matrix and shapefile.

    Args:
        district_id: District ID
        adjacency_dir: Directory containing voronoi output (adjacency matrices)
        building_path: Path to building shapefile
        normalize_features: Whether to normalize features
        scaler: Optional pre-fitted StandardScaler

    Returns:
        data: PyG Data object with node features, edges, and labels
        scaler: StandardScaler (fitted if not provided)
    """
    logger.debug(f"Loading district {district_id}...")

    # Load similarity/adjacency matrix
    sim_matrix_path = adjacency_dir / f"district_{district_id}_adjacency.pkl"

    if not sim_matrix_path.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")

    sim_matrix = pd.read_pickle(sim_matrix_path)
    logger.debug(f"Loaded similarity matrix: shape={sim_matrix.shape}")

    # Load building shapefile
    buildings_gdf = gpd.read_file(building_path)

    # Filter buildings to match those in similarity matrix
    building_ids_in_matrix = sim_matrix.index.tolist()

    # Create mapping from building ID to row index
    # Use 'id' field as the standard building identifier
    id_field = 'id'
    
    if id_field not in buildings_gdf.columns:
        logger.warning("No ID field found in buildings shapefile, using index")
        buildings_gdf['building_id'] = buildings_gdf.index
        id_field = 'building_id'

    # Handle type mismatch between adjacency matrix index and shapefile ID field
    # Adjacency matrix typically has int64 index, but shapefile might have float64
    if buildings_gdf[id_field].dtype in ['float64', 'float32']:
        # Convert matrix IDs to float for matching
        building_ids_in_matrix_typed = [float(bid) for bid in building_ids_in_matrix]
        logger.debug(f"Converting adjacency matrix IDs to float to match {id_field} type")
    else:
        # Keep as int
        building_ids_in_matrix_typed = building_ids_in_matrix
    
    # Filter to buildings in the matrix
    buildings_gdf = buildings_gdf[buildings_gdf[id_field].isin(building_ids_in_matrix_typed)].copy()

    # Sort to match matrix order (use typed IDs for mapping)
    buildings_gdf['_sort_key'] = buildings_gdf[id_field].map({bid: i for i, bid in enumerate(building_ids_in_matrix_typed)})
    buildings_gdf = buildings_gdf.sort_values('_sort_key').reset_index(drop=True)

    logger.debug(f"Filtered buildings: {len(buildings_gdf)} buildings")
    
    # Check if there's a mismatch between buildings and matrix
    if len(buildings_gdf) != len(sim_matrix):
        logger.warning(
            f"Mismatch between buildings ({len(buildings_gdf)}) and adjacency matrix ({len(sim_matrix)}). "
            f"Some buildings in the matrix may not exist in the shapefile."
        )
        # Filter the adjacency matrix to only include buildings that exist in shapefile
        # Need to convert back to original matrix index type (int)
        if buildings_gdf[id_field].dtype in ['float64', 'float32']:
            buildings_in_shapefile = [int(bid) for bid in buildings_gdf[id_field].tolist()]
        else:
            buildings_in_shapefile = buildings_gdf[id_field].tolist()
        
        mask = sim_matrix.index.isin(buildings_in_shapefile)
        sim_matrix = sim_matrix.loc[mask, mask]
        building_ids_in_matrix = sim_matrix.index.tolist()
        
        # Update typed list after filtering
        if buildings_gdf[id_field].dtype in ['float64', 'float32']:
            building_ids_in_matrix_typed = [float(bid) for bid in building_ids_in_matrix]
        else:
            building_ids_in_matrix_typed = building_ids_in_matrix
        
        logger.debug(f"Filtered adjacency matrix to {len(sim_matrix)} buildings")
        
        # Re-sort buildings to match filtered matrix
        buildings_gdf['_sort_key'] = buildings_gdf[id_field].map({bid: i for i, bid in enumerate(building_ids_in_matrix_typed)})
        buildings_gdf = buildings_gdf.sort_values('_sort_key').reset_index(drop=True)

    # Extract GAT features (now includes neighdis placeholder)
    features = extract_gat_features(buildings_gdf)

    # Calculate node degree (number of neighbors) for each building
    degrees = []
    for i in range(len(sim_matrix)):
        row = sim_matrix.iloc[i]
        num_neighbors = (row > 0).sum()  # Count non-zero entries
        degrees.append(num_neighbors)

    # Calculate average distance to neighbors from adjacency matrix
    avg_neighbor_distances = []
    for i in range(len(sim_matrix)):
        # Get non-zero distances for this building (its neighbors)
        row = sim_matrix.iloc[i]
        neighbor_distances = row[row > 0].values  # Get non-zero values

        if len(neighbor_distances) > 0:
            avg_dist = np.mean(neighbor_distances)
        else:
            avg_dist = 50.0  # Default value for isolated buildings

        avg_neighbor_distances.append(avg_dist)

    # Append degree and neighdis as additional features
    degree_column = np.array(degrees).reshape(-1, 1)
    neighdis_column = np.array(avg_neighbor_distances).reshape(-1, 1)
    features = np.concatenate([features, degree_column, neighdis_column], axis=1)

    logger.debug(f"Added degree feature: min={min(degrees)}, max={max(degrees)}, mean={np.mean(degrees):.2f}")
    logger.debug(f"Added neighdis feature: min={min(avg_neighbor_distances):.2f}, "
                f"max={max(avg_neighbor_distances):.2f}, mean={np.mean(avg_neighbor_distances):.2f}")

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

    # Extract labels if available (convert from 1-based to 0-based indexing)
    label_field = None
    for possible_label in ['label', 'class', 'category', 'cluster', 'type']:
        if possible_label in buildings_gdf.columns:
            label_field = possible_label
            break

    if label_field is not None:
        labels = buildings_gdf[label_field].values - 1  # Convert 1-based to 0-based
        y = torch.tensor(labels, dtype=torch.long)
        num_clusters = len(np.unique(labels))
        has_labels = True
        logger.debug(f"Found labels: {num_clusters} classes (converted from 1-based to 0-based)")
    else:
        # No labels available - inference mode
        y = torch.zeros(len(buildings_gdf), dtype=torch.long)
        num_clusters = 0  # Indicate no ground truth labels
        has_labels = False
        logger.debug(f"No label field found in district {district_id} (inference mode)")

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(buildings_gdf),
        district_id=district_id,
        num_clusters=torch.tensor([num_clusters], dtype=torch.float),
        has_labels=has_labels  # Flag to indicate if ground truth labels exist
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


def kfold_split(
    data_list: List[Data],
    n_splits: int = 5,
    random_seed: int = 42
) -> Iterator[Tuple[List[Data], List[Data]]]:
    """
    Create K-fold cross-validation splits.

    Args:
        data_list: List of PyG Data objects
        n_splits: Number of folds
        random_seed: Random seed for reproducibility

    Yields:
        train_data: List of training Data objects for this fold
        val_data: List of validation Data objects for this fold
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(data_list)):
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]

        logger.info(f"Fold {fold_idx + 1}/{n_splits}: {len(train_data)} train, {len(val_data)} val")

        yield train_data, val_data


def overlapping_cv_split(
    data_list: List[Data],
    n_splits: int = 5,
    val_ratio: float = 0.3,
    overlap_ratio: float = 0.15,
    random_seed: int = 42
) -> Iterator[Tuple[List[Data], List[Data]]]:
    """
    Create overlapping cross-validation splits (rolling window style).

    Compared to standard k-fold, this method:
    - Validation set has fixed size (e.g., 30%), rather than 1/k
    - Each fold overlaps with adjacent folds, increasing data utilization
    - Reduces fluctuation caused by too-small validation sets

    Args:
        data_list: List of PyG Data objects
        n_splits: Number of splits (folds)
        val_ratio: Validation set ratio (recommended 0.2-0.3)
        overlap_ratio: Overlap ratio between adjacent folds (recommended 0.1-0.2)
        random_seed: Random seed

    Yields:
        train_data: Training data list for this fold
        val_data: Validation data list for this fold

    Example:
        100 samples, n_splits=5, val_ratio=0.3, overlap_ratio=0.15
        - Fold 1: val[0:30],   train[30:100]  (70 training)
        - Fold 2: val[17:47],  train[47:100]+[0:17]  (70 training, 13 overlap)
        - Fold 3: val[34:64],  train[64:100]+[0:34]  (70 training, 13 overlap)
        - Fold 4: val[51:81],  train[81:100]+[0:51]  (70 training, 13 overlap)
        - Fold 5: val[68:98],  train[98:100]+[0:68]  (70 training, 13 overlap)
    """
    np.random.seed(random_seed)
    indices = np.arange(len(data_list))
    np.random.shuffle(indices)  # Shuffle order

    n = len(data_list)
    val_size = int(n * val_ratio)
    step_size = int(val_size * (1 - overlap_ratio))  # Step size for each move (considering overlap)

    logger.info(f"Overlapping CV: {n} samples, {n_splits} folds, val set {val_ratio*100:.1f}%, overlap {overlap_ratio*100:.1f}%")
    logger.info(f"Per fold: val set ~{val_size}, train set ~{n-val_size}, step size {step_size}")

    for fold_idx in range(n_splits):
        # Calculate validation set starting position for this fold (circular)
        val_start = (fold_idx * step_size) % n
        val_end = (val_start + val_size) % n

        # Handle circular boundary cases
        if val_end > val_start:
            # Normal case: val_indices are contiguous
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        else:
            # Cross boundary: val_indices split into two parts
            val_indices = np.concatenate([indices[val_start:], indices[:val_end]])
            train_indices = indices[val_end:val_start]

        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]

        logger.info(
            f"Fold {fold_idx + 1}/{n_splits}: "
            f"{len(train_data)} train samples, {len(val_data)} val samples "
            f"(val range: [{val_start}:{val_end}]{'cross-boundary' if val_end <= val_start else ''})"
        )

        yield train_data, val_data


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
    adjacency_dir: Path,
    building_path: Path,
    normalize_features: bool = True
) -> Tuple[List[Data], StandardScaler]:
    """
    Load multiple districts and fit a common scaler.

    Args:
        district_ids: List of district IDs
        adjacency_dir: Directory containing voronoi output
        building_path: Path to building shapefile
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
                district_id, adjacency_dir, building_path,
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

