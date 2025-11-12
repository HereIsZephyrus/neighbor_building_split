"""Compute building adjacency matrix from Voronoi diagrams."""

import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from ..utils.logger import get_logger

logger = get_logger(__name__)


def find_adjacent_voronoi_regions(voronoi_gdf: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    """Identify Voronoi polygons sharing boundaries using spatial indexing."""

    adjacent_pairs = []
    sindex = voronoi_gdf.sindex

    for _, row in voronoi_gdf.iterrows():
        building_id_i = row['building_id']
        geom_i = row.geometry

        possible_neighbors_idx = list(sindex.intersection(geom_i.bounds))
        possible_neighbors = voronoi_gdf.iloc[possible_neighbors_idx]

        for _, neighbor_row in possible_neighbors.iterrows():
            building_id_j = neighbor_row['building_id']

            if building_id_i == building_id_j or building_id_i >= building_id_j:
                continue

            geom_j = neighbor_row.geometry

            if geom_i.touches(geom_j):
                adjacent_pairs.append((building_id_i, building_id_j))
            elif geom_i.intersects(geom_j):
                intersection = geom_i.intersection(geom_j)
                if hasattr(intersection, 'length') and intersection.length > 0:
                    adjacent_pairs.append((building_id_i, building_id_j))

    logger.debug("Found %d adjacent pairs from %d buildings", 
                len(adjacent_pairs), len(voronoi_gdf))

    return adjacent_pairs


def compute_building_distance(building_i, building_j) -> float:
    """Calculate shortest distance between two building geometries."""
    return building_i.distance(building_j)


def create_building_id_mapping(buildings_gdf: gpd.GeoDataFrame) -> Dict[int, int]:
    """Create mapping from row index to building ID."""
    id_field = None
    for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'fid']:
        if possible_id in buildings_gdf.columns:
            id_field = possible_id
            break

    id_mapping = {}

    for idx, building in buildings_gdf.iterrows():
        if id_field is not None:
            building_id = building.get(id_field)
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
    Create symmetric adjacency matrix with distances between adjacent buildings.

    Matrix values: 0 = not adjacent, >0 = distance in meters.
    """
    adjacent_pairs = find_adjacent_voronoi_regions(voronoi_gdf)

    if len(adjacent_pairs) == 0:
        logger.warning("No adjacent pairs, returning zero matrix")
        building_ids = sorted(voronoi_gdf['building_id'].unique())
        return pd.DataFrame(0.0, index=building_ids, columns=building_ids)

    id_mapping = create_building_id_mapping(buildings_gdf)
    id_to_idx = {building_id: idx for idx, building_id in id_mapping.items()}
    building_ids = sorted(voronoi_gdf['building_id'].unique())

    matrix = pd.DataFrame(0.0, index=building_ids, columns=building_ids)

    distances_computed = 0
    skipped_missing = 0

    for building_id_i, building_id_j in adjacent_pairs:
        if building_id_i not in id_to_idx or building_id_j not in id_to_idx:
            skipped_missing += 1
            continue

        idx_i = id_to_idx[building_id_i]
        idx_j = id_to_idx[building_id_j]

        geom_i = buildings_gdf.loc[idx_i, 'geometry']
        geom_j = buildings_gdf.loc[idx_j, 'geometry']

        distance = compute_building_distance(geom_i, geom_j)

        matrix.loc[building_id_i, building_id_j] = distance
        matrix.loc[building_id_j, building_id_i] = distance

        distances_computed += 1

    if skipped_missing > 0:
        logger.warning("Skipped %d pairs with missing geometries", skipped_missing)

    non_zero_values = matrix.values[matrix.values > 0]
    if len(non_zero_values) > 0:
        logger.debug(
            "Adjacency: %d pairs, distances: %.2f-%.2f m (mean: %.2f)",
            len(non_zero_values) // 2,
            non_zero_values.min(),
            non_zero_values.max(),
            non_zero_values.mean()
        )
    else:
        logger.warning("Adjacency matrix empty")

    _verify_matrix_properties(matrix)

    return matrix


def _verify_matrix_properties(matrix: pd.DataFrame) -> None:
    """Verify adjacency matrix properties: zero diagonal, symmetric, non-negative."""
    diagonal = np.diag(matrix.values)
    if not np.allclose(diagonal, 0):
        logger.warning("Non-zero diagonal (max: %.2f)", diagonal.max())

    if not np.allclose(matrix.values, matrix.values.T):
        max_diff = np.abs(matrix.values - matrix.values.T).max()
        logger.warning("Matrix asymmetric (max diff: %.2e)", max_diff)

    if (matrix.values < 0).any():
        logger.warning("Matrix has negative values")

