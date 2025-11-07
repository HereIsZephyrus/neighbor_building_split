"""Building feature extraction from shapefiles."""

import numpy as np
import geopandas as gpd
from typing import List

from .logger import get_logger

logger = get_logger()


def extract_gat_features(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract GAT features by reading columns directly from shapefile.
    Note: Node degree is added as the 5th feature during graph construction.

    GAT Features (4 base features):
    1. height: Building height
    2. albedo: Surface albedo
    3. hwratio: Height-to-width ratio
    4. density: Building density metric
    5. degree: Number of neighboring buildings (added during graph construction)

    Args:
        buildings_gdf: GeoDataFrame with building attributes

    Returns:
        numpy array of shape (N, 4) with extracted features
        (degree feature is added later in BuildingGraphDataset)
    """
    n_buildings = len(buildings_gdf)
    required_columns = ['height', 'albedo', 'hwratio', 'density']
    
    # Validate required columns exist
    missing_columns = [col for col in required_columns if col not in buildings_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for GAT features: {missing_columns}")
    
    logger.debug("Extracting GAT features for %d buildings...", n_buildings)
    
    # Extract features directly from columns
    features = buildings_gdf[required_columns].values.astype(np.float64)
    
    # Handle NaN and Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.debug("GAT feature extraction complete. Shape: %s", features.shape)
    logger.debug("Feature ranges - min: %s, max: %s", features.min(axis=0), features.max(axis=0))
    
    return features


def extract_clustering_features(buildings_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract clustering features by reading columns directly from shapefile.
    Note: Node degree is added as the 13th feature during clustering.

    Clustering Features (12 base features):
    1. height: Building height
    2. area: Building footprint area
    3. perimeter: Building perimeter
    4. orientation: Main axis orientation
    5. elongation: Height-to-width ratio
    6. concavity: Concavity measure
    7. circularity: Circularity measure
    8. radius: Distance from center
    9. factality: Fractal dimension
    10. overlap: Overlap metric
    11. shape: Shape metric
    12. density: Building density
    13. degree: Number of neighboring buildings (added during clustering)

    Args:
        buildings_gdf: GeoDataFrame with building attributes

    Returns:
        numpy array of shape (N, 12) with extracted features
        (degree feature is added later during clustering)
    """
    n_buildings = len(buildings_gdf)
    required_columns = [
        'height', 'area', 'perimeter', 'orientation', 'elongation',
        'concavity', 'circularity', 'radius', 'factality', 'overlap',
        'shape', 'density'
    ]
    
    # Validate required columns exist
    missing_columns = [col for col in required_columns if col not in buildings_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for clustering features: {missing_columns}")
    
    logger.debug("Extracting clustering features for %d buildings...", n_buildings)
    
    # Extract features directly from columns
    features = buildings_gdf[required_columns].values.astype(np.float64)
    
    # Handle NaN and Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.debug("Clustering feature extraction complete. Shape: %s", features.shape)
    logger.debug("Feature ranges - min: %s, max: %s", features.min(axis=0), features.max(axis=0))
    
    return features


def get_gat_feature_names() -> List[str]:
    """Return list of GAT feature names."""
    return ['height', 'albedo', 'hwratio', 'density', 'degree']


def get_clustering_feature_names() -> List[str]:
    """Return list of clustering feature names."""
    return [
        'height', 'area', 'perimeter', 'orientation', 'elongation',
        'concavity', 'circularity', 'radius', 'factality', 'overlap',
        'shape', 'density', 'degree'
    ]

