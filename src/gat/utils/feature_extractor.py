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
