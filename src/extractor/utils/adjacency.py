"""Adjacency matrix computation for building neighborhoods based on Voronoi diagrams."""

import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from ..utils.logger import get_logger

logger = get_logger()


def find_adjacent_voronoi_regions(voronoi_gdf: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    """
    Identify which Voronoi polygons share boundaries.

    Args:
        voronoi_gdf: GeoDataFrame with Voronoi polygons and building_id column

    Returns:
        List of tuples (building_id_i, building_id_j) for adjacent pairs
    """
    logger.debug("Finding adjacent Voronoi regions among %d polygons", len(voronoi_gdf))

    adjacent_pairs = []

    # Create a spatial index for efficient neighbor search
    sindex = voronoi_gdf.sindex

    for _, row in voronoi_gdf.iterrows():
        building_id_i = row['building_id']
        geom_i = row.geometry

        # Find potential neighbors using spatial index
        possible_neighbors_idx = list(sindex.intersection(geom_i.bounds))
        possible_neighbors = voronoi_gdf.iloc[possible_neighbors_idx]

        for _, neighbor_row in possible_neighbors.iterrows():
            building_id_j = neighbor_row['building_id']

            # Skip self
            if building_id_i == building_id_j:
                continue

            # Only process each pair once (avoid duplicates)
            if building_id_i >= building_id_j:
                continue

            geom_j = neighbor_row.geometry

            # Check if they share a boundary (touch or intersect with non-zero length)
            if geom_i.touches(geom_j):
                adjacent_pairs.append((building_id_i, building_id_j))
            elif geom_i.intersects(geom_j):
                intersection = geom_i.intersection(geom_j)
                # Check if intersection is a line (has length > 0)
                if hasattr(intersection, 'length') and intersection.length > 0:
                    adjacent_pairs.append((building_id_i, building_id_j))

    logger.info("Found %d adjacent pairs among %d buildings", 
                len(adjacent_pairs), len(voronoi_gdf))

    return adjacent_pairs


def compute_building_distance(building_i, building_j) -> float:
    """
    Calculate shortest distance between two building geometries.

    Args:
        building_i: Shapely geometry of first building
        building_j: Shapely geometry of second building

    Returns:
        Distance in meters (CRS units)
    """
    return building_i.distance(building_j)


def create_building_id_mapping(buildings_gdf: gpd.GeoDataFrame) -> Dict[int, int]:
    """
    Create mapping from building row index to building ID.

    Args:
        buildings_gdf: GeoDataFrame with building geometries

    Returns:
        Dictionary mapping row index to building ID
    """
    id_field = None
    for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'fid']:
        if possible_id in buildings_gdf.columns:
            id_field = possible_id
            break

    id_mapping = {}

    for idx, building in buildings_gdf.iterrows():
        if id_field is not None:
            building_id = building.get(id_field)
            # Handle NULL/NaN/None values
            if building_id is None or (isinstance(building_id, float) and math.isnan(building_id)):
                building_id = int(idx) + 1 if isinstance(idx, int) else hash(str(idx)) % 2147483647
            else:
                building_id = int(building_id)
        else:
            building_id = int(idx) + 1 if isinstance(idx, int) else hash(str(idx)) % 2147483647

        id_mapping[idx] = building_id

    return id_mapping


def create_adjacency_matrix(
    voronoi_gdf: gpd.GeoDataFrame, 
    buildings_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Create adjacency matrix with shortest distances between adjacent buildings.

    The matrix uses building IDs as both index and columns. Values represent:
    - 0: Buildings are not adjacent
    - >0: Shortest distance between adjacent buildings in meters

    Args:
        voronoi_gdf: GeoDataFrame with Voronoi polygons and building_id column
        buildings_gdf: GeoDataFrame with building geometries

    Returns:
        Pandas DataFrame with building_ids as index and columns, containing distances
    """
    logger.info("Creating adjacency matrix for %d buildings", len(voronoi_gdf))

    # Find adjacent pairs based on Voronoi diagram
    adjacent_pairs = find_adjacent_voronoi_regions(voronoi_gdf)

    if len(adjacent_pairs) == 0:
        logger.warning("No adjacent pairs found, returning empty matrix")
        building_ids = sorted(voronoi_gdf['building_id'].unique())
        return pd.DataFrame(0.0, index=building_ids, columns=building_ids)

    # Create building ID to geometry mapping
    id_mapping = create_building_id_mapping(buildings_gdf)

    # Create reverse mapping: building_id -> row index
    id_to_idx = {building_id: idx for idx, building_id in id_mapping.items()}

    # Get all unique building IDs from Voronoi diagram
    building_ids = sorted(voronoi_gdf['building_id'].unique())

    # Initialize matrix with zeros
    matrix = pd.DataFrame(0.0, index=building_ids, columns=building_ids)

    # Compute distances for adjacent pairs
    logger.debug("Computing distances for %d adjacent pairs", len(adjacent_pairs))

    distances_computed = 0
    skipped_missing = 0

    for building_id_i, building_id_j in adjacent_pairs:
        # Get building geometries from buildings_gdf
        if building_id_i not in id_to_idx or building_id_j not in id_to_idx:
            skipped_missing += 1
            continue

        idx_i = id_to_idx[building_id_i]
        idx_j = id_to_idx[building_id_j]

        geom_i = buildings_gdf.loc[idx_i, 'geometry']
        geom_j = buildings_gdf.loc[idx_j, 'geometry']

        # Compute shortest distance
        distance = compute_building_distance(geom_i, geom_j)

        # Set symmetric values in matrix
        matrix.loc[building_id_i, building_id_j] = distance
        matrix.loc[building_id_j, building_id_i] = distance

        distances_computed += 1

    if skipped_missing > 0:
        logger.warning("Skipped %d adjacent pairs due to missing building geometries", 
                      skipped_missing)

    # Log statistics
    non_zero_values = matrix.values[matrix.values > 0]
    if len(non_zero_values) > 0:
        logger.info(
            "Adjacency matrix created: shape=%s, adjacencies=%d, "
            "distance stats: min=%.2f, max=%.2f, mean=%.2f meters",
            matrix.shape, 
            len(non_zero_values) // 2,  # Divide by 2 because matrix is symmetric
            non_zero_values.min(),
            non_zero_values.max(),
            non_zero_values.mean()
        )
    else:
        logger.warning("Adjacency matrix has no non-zero values")

    # Verify matrix properties
    _verify_matrix_properties(matrix)

    return matrix


def _verify_matrix_properties(matrix: pd.DataFrame) -> None:
    """
    Verify that the adjacency matrix has expected properties.

    Args:
        matrix: Adjacency matrix to verify
    """
    # Check diagonal is zero
    diagonal = np.diag(matrix.values)
    if not np.allclose(diagonal, 0):
        logger.warning("Matrix diagonal is not all zeros (max: %.2f)", diagonal.max())

    # Check symmetry
    if not np.allclose(matrix.values, matrix.values.T):
        logger.warning("Matrix is not symmetric")
        max_diff = np.abs(matrix.values - matrix.values.T).max()
        logger.warning("Maximum asymmetry: %.2e", max_diff)
    else:
        logger.debug("Matrix symmetry verified")

    # Check for negative values
    if (matrix.values < 0).any():
        logger.warning("Matrix contains negative values")

