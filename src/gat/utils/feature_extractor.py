"""Building feature extraction from shapefiles."""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from typing import Optional
import warnings

from .logger import get_logger

logger = get_logger()


def extract_building_features(
    buildings_gdf: gpd.GeoDataFrame,
    normalize_spatial: bool = True
) -> np.ndarray:
    """
    Extract 13 geometric and shape features from building geometries.
    Note: Node degree is added as the 14th feature during graph construction.

    Features extracted:
    1. area: Building footprint area
    2. perimeter: Building perimeter
    3. bounds_width: Width of bounding box
    4. bounds_height: Height of bounding box
    5. compactness: 4π * area / perimeter² (circle = 1.0)
    6. elongation: bounds_height / bounds_width
    7. rectangularity: area / bounding_box_area
    8. orientation_angle: Main axis orientation (radians)
    9. convexity: area / convex_hull_area
    10. num_vertices: Number of polygon vertices
    11. perimeter_area_ratio: perimeter / sqrt(area)
    12. centroid_distance: Distance from district center (normalized)
    13. floor: Mean floor of the building
    14. degree: Number of neighboring buildings (added during graph construction)

    Args:
        buildings_gdf: GeoDataFrame with building geometries
        normalize_spatial: If True, normalize centroid coordinates within district bounds

    Returns:
        numpy array of shape (N, 13) with extracted features
        (degree feature is added later in BuildingGraphDataset)
    """
    n_buildings = len(buildings_gdf)
    features = np.zeros((n_buildings, 13))

    # Compute district bounds for spatial normalization
    if normalize_spatial:
        district_bounds = buildings_gdf.total_bounds  # minx, miny, maxx, maxy
        district_center_x = (district_bounds[0] + district_bounds[2]) / 2
        district_center_y = (district_bounds[1] + district_bounds[3]) / 2
        district_width = district_bounds[2] - district_bounds[0]
        district_height = district_bounds[3] - district_bounds[1]
        district_scale = max(district_width, district_height)

    logger.debug(f"Extracting features for {n_buildings} buildings...")

    for idx, (_, building) in enumerate(buildings_gdf.iterrows()):
        geom = building.geometry

        # Skip invalid geometries
        if geom is None or geom.is_empty:
            logger.warning(f"Building {idx} has invalid geometry, using zero features")
            continue

        try:
            # Feature 1: Area
            area = geom.area
            features[idx, 0] = area

            # Feature 2: Perimeter
            perimeter = geom.length
            features[idx, 1] = perimeter

            # Feature 3-4: Bounding box dimensions
            bounds = geom.bounds  # minx, miny, maxx, maxy
            bounds_width = bounds[2] - bounds[0]
            bounds_height = bounds[3] - bounds[1]
            features[idx, 2] = bounds_width
            features[idx, 3] = bounds_height

            # Feature 5: Compactness (circularity)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
                features[idx, 4] = min(compactness, 1.0)  # Cap at 1.0

            # Feature 6: Elongation
            if bounds_width > 0:
                elongation = bounds_height / bounds_width
                features[idx, 5] = elongation
            else:
                features[idx, 5] = 1.0

            # Feature 7: Rectangularity
            bbox_area = bounds_width * bounds_height
            if bbox_area > 0:
                rectangularity = area / bbox_area
                features[idx, 6] = rectangularity

            # Feature 8: Orientation angle
            # Use minimum rotated rectangle
            try:
                min_rect = geom.minimum_rotated_rectangle
                if min_rect.geom_type == 'Polygon':
                    coords = list(min_rect.exterior.coords)
                    if len(coords) >= 3:
                        # Calculate angle of first edge
                        dx = coords[1][0] - coords[0][0]
                        dy = coords[1][1] - coords[0][1]
                        angle = np.arctan2(dy, dx)
                        # Normalize to [0, π/2]
                        angle = abs(angle) % (np.pi / 2)
                        features[idx, 7] = angle
            except Exception as e:
                logger.debug(f"Could not compute orientation for building {idx}: {e}")
                features[idx, 7] = 0.0

            # Feature 9: Convexity
            try:
                convex_hull = geom.convex_hull
                convex_area = convex_hull.area
                if convex_area > 0:
                    convexity = area / convex_area
                    features[idx, 8] = convexity
                else:
                    features[idx, 8] = 1.0
            except Exception as e:
                logger.debug(f"Could not compute convexity for building {idx}: {e}")
                features[idx, 8] = 1.0

            # Feature 10: Number of vertices
            if geom.geom_type == 'Polygon':
                num_vertices = len(geom.exterior.coords) - 1  # Exclude closing point
                features[idx, 9] = num_vertices
            elif geom.geom_type == 'MultiPolygon':
                num_vertices = sum(len(poly.exterior.coords) - 1 for poly in geom.geoms)
                features[idx, 9] = num_vertices

            # Feature 11: Perimeter to area ratio
            if area > 0:
                perimeter_area_ratio = perimeter / np.sqrt(area)
                features[idx, 10] = perimeter_area_ratio

            # Feature 12: Distance from district center (normalized)
            if normalize_spatial:
                centroid = geom.centroid
                dist_x = centroid.x - district_center_x
                dist_y = centroid.y - district_center_y
                distance = np.sqrt(dist_x**2 + dist_y**2)
                normalized_distance = distance / district_scale if district_scale > 0 else 0
                features[idx, 11] = normalized_distance

        except Exception as e:
            logger.warning(f"Error extracting features for building {idx}: {e}")
            continue

    # Handle NaN and Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(f"Feature extraction complete. Shape: {features.shape}")
    logger.debug(f"Feature ranges - min: {features.min(axis=0)}, max: {features.max(axis=0)}")

    return features


def get_feature_names() -> list:
    """Return list of feature names."""
    return [
        'area',
        'perimeter',
        'bounds_width',
        'bounds_height',
        'compactness',
        'elongation',
        'rectangularity',
        'orientation_angle',
        'convexity',
        'num_vertices',
        'perimeter_area_ratio',
        'centroid_distance',
        'floor',
        'degree'
    ]

