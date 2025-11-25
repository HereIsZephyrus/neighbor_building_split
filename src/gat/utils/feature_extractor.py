"""Building feature extraction from shapefiles.

This module implements a two-stage feature extraction strategy:
1. GAT features: Discriminative features for building classification
2. Clustering features: Morphological features for spatial grouping

Design Rationale:
- GAT focuses on "what type of building?" using discriminative features
- Spectral clustering focuses on "which buildings belong together?" using morphological similarity
- This separation allows task-specific feature selection for better performance
"""

import numpy as np
import geopandas as gpd
from typing import Tuple, Optional
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler

from .logger import get_logger

logger = get_logger(__name__)

# Cache for feature configuration
_FEATURE_CONFIG = None


def load_feature_config() -> dict:
    """
    Load feature configuration from features_config.yaml.

    Returns:
        Dictionary containing GAT features, clustering features, and descriptions
    """
    global _FEATURE_CONFIG

    if _FEATURE_CONFIG is not None:
        return _FEATURE_CONFIG

    # Try to find config file
    config_path = Path(__file__).parent.parent / 'features_config.yaml'

    if not config_path.exists():
        logger.warning(f"Feature config file not found at {config_path}, using defaults")
        # Return default configuration (using shapefile column names)
        # Note: degree and neighdis are added dynamically during graph construction
        _FEATURE_CONFIG = {
            'gat_features': ['height', 'albedo', 'hwratio'],
            'clustering_features': [
                'height', 'area', 'perimeter', 'orientatio', 'elongation',
                'concavity', 'circularit', 'rectangula', 'fractality',
                'rangeIndex'
            ]
        }
        return _FEATURE_CONFIG

    with open(config_path, 'r', encoding='utf-8') as f:
        _FEATURE_CONFIG = yaml.safe_load(f)

    logger.debug(f"Loaded feature configuration from {config_path}")
    return _FEATURE_CONFIG


def extract_gat_features(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract GAT features for building classification.

    These features are selected for their discriminative power in predicting
    building labels (e.g., residential, commercial, industrial).

    Note: Node degree and neighdis are added as additional features during graph construction.

    Design Choice:
    - Uses discriminative features that help GAT distinguish between building types
    - Features are loaded from features_config.yaml for easy configuration
    - Default: height, albedo, hwratio (plus degree and neighdis added later)

    Args:
        buildings_gdf: GeoDataFrame with building attributes

    Returns:
        numpy array of shape (N, num_features) with extracted features
        Note: degree and neighdis features are added later in BuildingGraphDataset

    Raises:
        ValueError: If required feature columns are missing from the GeoDataFrame
    """
    n_buildings = len(buildings_gdf)

    # Load feature names from config
    config = load_feature_config()
    required_columns = config.get('gat_features', ['height', 'albedo', 'hwratio'])

    # Validate required columns exist
    missing_columns = [col for col in required_columns if col not in buildings_gdf.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for GAT features: {missing_columns}. "
            f"Available columns: {list(buildings_gdf.columns)}"
        )

    logger.debug("Extracting GAT features for %d buildings (features: %s)...", 
                 n_buildings, required_columns)

    # Extract features directly from columns
    features = buildings_gdf[required_columns].values.astype(np.float64)

    # Handle NaN and Inf values (replace with 0 for robustness)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug("GAT feature extraction complete. Shape: %s", features.shape)
    logger.debug("Feature ranges - min: %s, max: %s", features.min(axis=0), features.max(axis=0))

    return features


def extract_clustering_features(
    buildings_gdf: gpd.GeoDataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Extract morphological features for spectral clustering.

    These features are selected for their ability to capture spatial and morphological
    similarity, enabling spatial grouping of buildings with similar characteristics.

    Design Choice:
    - Uses morphological features (area, shape, orientation) for spatial grouping
    - Different from GAT features: focuses on "which buildings belong together"
    - GAT focuses on "what type of building" using discriminative features
    - This separation allows spectral clustering to smooth GAT predictions spatially
    - Features are standardized to ensure all morphological attributes contribute equally

    Typical Features:
    - Geometric: area, perimeter, orientation, elongation
    - Shape: concavity, circularity, shape complexity
    - Spatial: radius (from center), overlap metrics
    - Common: height, density (shared with GAT for consistency)

    Note: Node degree can be added during clustering if needed.

    Args:
        buildings_gdf: GeoDataFrame with building attributes
        scaler: Optional pre-fitted StandardScaler for normalization
        fit_scaler: If True, fit a new scaler on the features (for training)

    Returns:
        features: numpy array of shape (N, num_features) with extracted and standardized features
        scaler: Fitted or provided StandardScaler (None if no standardization applied)

    Raises:
        ValueError: If required feature columns are missing from the GeoDataFrame
    """
    n_buildings = len(buildings_gdf)

    # Load feature names from config
    config = load_feature_config()
    required_columns = config.get('clustering_features', [
        'height', 'area', 'perimeter', 'orientatio', 'elongation',
        'concavity', 'circularit', 'rectangula', 'fractality',
        'rangeIndex'
    ])

    # Validate required columns exist
    missing_columns = [col for col in required_columns if col not in buildings_gdf.columns]
    if missing_columns:
        # Provide helpful error message with available columns
        available_cols = [c for c in buildings_gdf.columns if c not in ['geometry', 'fid', 'OBJECTID', 'id']]
        raise ValueError(
            f"Missing required columns for clustering features: {missing_columns}. "
            f"Available feature columns: {available_cols}"
        )

    logger.debug("Extracting clustering features for %d buildings (features: %s)...", 
                 n_buildings, required_columns)

    # Extract features directly from columns
    features = buildings_gdf[required_columns].values.astype(np.float64)

    # Handle NaN and Inf values (replace with 0 for robustness)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug("Clustering feature extraction complete (before normalization). Shape: %s", features.shape)
    logger.debug("Feature ranges (raw) - min: %s, max: %s", features.min(axis=0), features.max(axis=0))

    # Standardize features if requested
    if fit_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        logger.debug("Fitted new StandardScaler on clustering features")
        logger.debug("Feature ranges (normalized) - mean: %s, std: %s", 
                     scaler.mean_, scaler.scale_)
    elif scaler is not None:
        features = scaler.transform(features)
        logger.debug("Applied existing StandardScaler to clustering features")
    else:
        logger.warning("No standardization applied to clustering features (scaler=None, fit_scaler=False)")

    return features, scaler


def fit_global_clustering_scaler(
    buildings_path: Path,
    adjacency_dir: Optional[Path] = None,
    district_ids: Optional[list] = None,
    sample_size: Optional[int] = None
) -> StandardScaler:
    """
    Fit a global StandardScaler for clustering features from all buildings.
    
    This function ensures consistent feature normalization across all districts
    by fitting the scaler on the entire building dataset (or a representative sample).
    
    Design Rationale:
    - Training and inference use the same scaler for consistency
    - Prevents order-dependency (first district determining normalization)
    - Provides representative statistics from the full population
    - Matches the approach used for GAT features in BuildingGraphDataset
    
    Args:
        buildings_path: Path to building shapefile containing all buildings
        adjacency_dir: Optional directory containing adjacency matrices (for filtering)
        district_ids: Optional list of district IDs to include (for filtering)
        sample_size: Optional number of buildings to sample (for large datasets)
    
    Returns:
        Fitted StandardScaler for clustering features
        
    Raises:
        ValueError: If required columns are missing
        FileNotFoundError: If buildings file doesn't exist
    """
    import pandas as pd
    
    logger.info("Fitting global clustering scaler from %s", buildings_path)
    
    # Load all buildings
    if not Path(buildings_path).exists():
        raise FileNotFoundError(f"Building shapefile not found: {buildings_path}")
    
    buildings_gdf = gpd.read_file(buildings_path)
    logger.info("Loaded %d buildings from %s", len(buildings_gdf), buildings_path)
    
    # Filter by district IDs if provided
    if district_ids is not None and adjacency_dir is not None:
        logger.info("Filtering buildings by %d districts...", len(district_ids))
        all_building_ids = set()
        
        for district_id in district_ids:
            adjacency_path = Path(adjacency_dir) / f"district_{district_id}_adjacency.pkl"
            if adjacency_path.exists():
                try:
                    adjacency_matrix = pd.read_pickle(adjacency_path)
                    building_ids = adjacency_matrix.index.tolist()
                    all_building_ids.update(building_ids)
                except Exception as e:
                    logger.warning("Failed to load adjacency for district %d: %s", district_id, e)
                    continue
        
        # Filter buildings
        id_field = 'id'
        if id_field in buildings_gdf.columns:
            # Handle type conversion
            if buildings_gdf[id_field].dtype in ['float64', 'float32']:
                all_building_ids_typed = [float(bid) for bid in all_building_ids]
            else:
                all_building_ids_typed = list(all_building_ids)
            
            buildings_gdf = buildings_gdf[buildings_gdf[id_field].isin(all_building_ids_typed)].copy()
            logger.info("Filtered to %d buildings in training districts", len(buildings_gdf))
    
    # Sample if requested (for very large datasets)
    if sample_size is not None and len(buildings_gdf) > sample_size:
        logger.info("Sampling %d buildings from %d total", sample_size, len(buildings_gdf))
        buildings_gdf = buildings_gdf.sample(n=sample_size, random_state=42)
    
    # Extract clustering features WITHOUT fitting scaler yet
    features, _ = extract_clustering_features(
        buildings_gdf,
        scaler=None,
        fit_scaler=False
    )
    
    # Now fit the scaler on all features
    scaler = StandardScaler()
    scaler.fit(features)
    
    logger.info("Fitted global clustering scaler on %d buildings", len(buildings_gdf))
    logger.info("Feature means: %s", scaler.mean_)
    logger.info("Feature stds: %s", scaler.scale_)
    
    return scaler
