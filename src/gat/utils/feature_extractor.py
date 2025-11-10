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
from typing import List
from pathlib import Path
import yaml

from .logger import get_logger

logger = get_logger()

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
        # Return default configuration
        _FEATURE_CONFIG = {
            'gat_features': ['height', 'albedo', 'hwratio', 'density'],
            'clustering_features': [
                'height', 'area', 'perimeter', 'orientation', 'elongation',
                'concavity', 'circularity', 'radius', 'factality', 'overlap',
                'shape', 'density'
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

    Note: Node degree is added as an additional feature during graph construction.

    Design Choice:
    - Uses discriminative features that help GAT distinguish between building types
    - Features are loaded from features_config.yaml for easy configuration
    - Default: height, albedo, hwratio, density (plus degree added later)

    Args:
        buildings_gdf: GeoDataFrame with building attributes

    Returns:
        numpy array of shape (N, num_features) with extracted features
        Note: degree feature is added later in BuildingGraphDataset

    Raises:
        ValueError: If required feature columns are missing from the GeoDataFrame
    """
    n_buildings = len(buildings_gdf)

    # Load feature names from config
    config = load_feature_config()
    required_columns = config.get('gat_features', ['height', 'albedo', 'hwratio', 'density'])

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


def extract_clustering_features(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract morphological features for spectral clustering.

    These features are selected for their ability to capture spatial and morphological
    similarity, enabling spatial grouping of buildings with similar characteristics.

    Design Choice:
    - Uses morphological features (area, shape, orientation) for spatial grouping
    - Different from GAT features: focuses on "which buildings belong together"
    - GAT focuses on "what type of building" using discriminative features
    - This separation allows spectral clustering to smooth GAT predictions spatially

    Typical Features:
    - Geometric: area, perimeter, orientation, elongation
    - Shape: concavity, circularity, shape complexity
    - Spatial: radius (from center), overlap metrics
    - Common: height, density (shared with GAT for consistency)

    Note: Node degree can be added during clustering if needed.

    Args:
        buildings_gdf: GeoDataFrame with building attributes

    Returns:
        numpy array of shape (N, num_features) with extracted features

    Raises:
        ValueError: If required feature columns are missing from the GeoDataFrame
    """
    n_buildings = len(buildings_gdf)

    # Load feature names from config
    config = load_feature_config()
    required_columns = config.get('clustering_features', [
        'height', 'area', 'perimeter', 'orientation', 'elongation',
        'concavity', 'circularity', 'radius', 'factality', 'overlap',
        'shape', 'density'
    ])

    # Validate required columns exist
    missing_columns = [col for col in required_columns if col not in buildings_gdf.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for clustering features: {missing_columns}. "
            f"Available columns: {list(buildings_gdf.columns)}"
        )

    logger.debug("Extracting clustering features for %d buildings (features: %s)...", 
                 n_buildings, required_columns)

    # Extract features directly from columns
    features = buildings_gdf[required_columns].values.astype(np.float64)

    # Handle NaN and Inf values (replace with 0 for robustness)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug("Clustering feature extraction complete. Shape: %s", features.shape)
    logger.debug("Feature ranges - min: %s, max: %s", features.min(axis=0), features.max(axis=0))

    return features


def get_gat_feature_names() -> List[str]:
    """
    Get list of GAT feature names from configuration.

    Returns:
        List of feature names used in GAT training
    """
    config = load_feature_config()
    feature_names = config.get('gat_features', ['height', 'albedo', 'hwratio', 'density'])
    # Add degree if not already present (it's added during graph construction)
    if 'degree' not in feature_names:
        feature_names = feature_names + ['degree']
    return feature_names


def get_clustering_feature_names() -> List[str]:
    """
    Get list of clustering feature names from configuration.

    Returns:
        List of feature names used in spectral clustering
    """
    config = load_feature_config()
    feature_names = config.get('clustering_features', [
        'height', 'area', 'perimeter', 'orientation', 'elongation',
        'concavity', 'circularity', 'radius', 'factality', 'overlap',
        'shape', 'density'
    ])
    # Optionally add degree during clustering if needed
    return feature_names
