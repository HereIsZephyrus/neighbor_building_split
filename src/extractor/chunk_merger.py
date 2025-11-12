"""Merge spatial chunks into unified district outputs."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from affine import Affine
from .utils.logger import get_logger
from .utils.adjacency import find_adjacent_voronoi_regions, compute_building_distance, create_building_id_mapping

logger = get_logger(__name__)


def merge_voronoi_chunks(
    chunk_voronoi_gdfs: List[gpd.GeoDataFrame],
    overlap_width: int = 100
) -> gpd.GeoDataFrame:
    """
    Merge Voronoi polygons by building ID, unifying split regions.

    Returns unified GeoDataFrame with one row per building.
    """
    logger.debug("Merging Voronoi polygons from %d chunks", len(chunk_voronoi_gdfs))

    if len(chunk_voronoi_gdfs) == 0:
        logger.warning("No chunks to merge")
        return gpd.GeoDataFrame()

    all_voronoi = pd.concat(chunk_voronoi_gdfs, ignore_index=True)

    if len(all_voronoi) == 0:
        logger.warning("No polygons to merge")
        return gpd.GeoDataFrame()

    logger.debug("Merging %d polygons", len(all_voronoi))

    merged_rows = []

    for building_id, group in all_voronoi.groupby('building_id'):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
        else:
            merged_geom = unary_union(group.geometry.tolist())
            row_dict = group.iloc[0].to_dict()
            row_dict['geometry'] = merged_geom
            row_dict['area'] = merged_geom.area
            merged_rows.append(row_dict)

    merged_gdf = gpd.GeoDataFrame(merged_rows, crs=all_voronoi.crs)

    logger.info("Merged %d polygons into %d buildings (%.2f m²)",
                len(all_voronoi), len(merged_gdf), merged_gdf['area'].sum())

    return merged_gdf


def merge_adjacency_matrices(
    chunk_adjacencies: List[pd.DataFrame],
    merged_voronoi_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Merge and re-validate adjacency matrices from chunks.

    Recalculates relationships using merged Voronoi geometries to ensure
    accuracy at chunk boundaries.
    """
    logger.debug("Merging adjacency matrices from %d chunks", len(chunk_adjacencies))

    if len(chunk_adjacencies) == 0:
        logger.warning("No adjacencies to merge")
        return pd.DataFrame()

    all_building_ids = sorted(merged_voronoi_gdf['building_id'].unique())
    combined_matrix = pd.DataFrame(0.0, index=all_building_ids, columns=all_building_ids)

    for chunk_adj in chunk_adjacencies:
        for idx in chunk_adj.index:
            if idx not in combined_matrix.index:
                continue
            for col in chunk_adj.columns:
                if col not in combined_matrix.columns:
                    continue

                chunk_val = chunk_adj.loc[idx, col]
                current_val = combined_matrix.loc[idx, col]

                if chunk_val > 0:
                    if current_val == 0:
                        combined_matrix.loc[idx, col] = chunk_val
                    else:
                        combined_matrix.loc[idx, col] = min(current_val, chunk_val)

    logger.debug("Combined matrix: %d adjacency pairs", (combined_matrix.values > 0).sum() // 2)

    logger.debug("Re-validating adjacencies on merged geometries...")

    adjacent_pairs = find_adjacent_voronoi_regions(merged_voronoi_gdf)

    id_mapping = create_building_id_mapping(buildings_gdf)
    id_to_idx = {building_id: idx for idx, building_id in id_mapping.items()}

    final_matrix = pd.DataFrame(0.0, index=all_building_ids, columns=all_building_ids)

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

        final_matrix.loc[building_id_i, building_id_j] = distance
        final_matrix.loc[building_id_j, building_id_i] = distance

        distances_computed += 1

    if skipped_missing > 0:
        logger.warning("Skipped %d pairs with missing geometries", skipped_missing)

    non_zero_values = final_matrix.values[final_matrix.values > 0]
    if len(non_zero_values) > 0:
        logger.info(
            "Adjacency matrix: %d pairs, distances: %.2f-%.2f m (mean: %.2f)",
            len(non_zero_values) // 2,
            non_zero_values.min(),
            non_zero_values.max(),
            non_zero_values.mean()
        )
    else:
        logger.warning("Adjacency matrix has no non-zero values")

    return final_matrix


def stitch_rasters(
    chunk_rasters: List[np.ndarray],
    chunk_transforms: List[Affine],
    chunk_bounds: List[Tuple[float, float, float, float]],
    full_bounds: Tuple[float, float, float, float],
    full_transform: Affine,
    full_shape: Tuple[int, int],
    overlap: int = 100  # noqa: ARG001 - kept for API consistency
) -> np.ndarray:
    """
    Stitch chunk rasters into unified output.

    Uses priority map to handle overlapping regions.
    """
    logger.debug("Stitching %d rasters (shape: %s)", len(chunk_rasters), full_shape)

    full_raster = np.full(full_shape, -999, dtype=np.int32)
    priority_map = np.zeros(full_shape, dtype=np.int32)

    for chunk_idx, (chunk_raster, _chunk_transform, chunk_bounds) in enumerate(
        zip(chunk_rasters, chunk_transforms, chunk_bounds)
    ):
        chunk_minx, _chunk_miny, _chunk_maxx, chunk_maxy = chunk_bounds

        col_start = int((chunk_minx - full_bounds[0]) / full_transform.a)
        row_start = int((full_bounds[3] - chunk_maxy) / abs(full_transform.e))

        chunk_h, chunk_w = chunk_raster.shape
        col_end = min(col_start + chunk_w, full_shape[1])
        row_end = min(row_start + chunk_h, full_shape[0])

        col_start = max(0, col_start)
        row_start = max(0, row_start)

        chunk_col_start = max(0, -int((chunk_minx - full_bounds[0]) / full_transform.a))
        chunk_row_start = max(0, -int((full_bounds[3] - chunk_maxy) / abs(full_transform.e)))
        chunk_col_end = chunk_col_start + (col_end - col_start)
        chunk_row_end = chunk_row_start + (row_end - row_start)

        chunk_slice = chunk_raster[chunk_row_start:chunk_row_end, chunk_col_start:chunk_col_end]
        chunk_priority = np.ones_like(chunk_slice, dtype=np.int32) * (chunk_idx + 1)

        mask = (priority_map[row_start:row_end, col_start:col_end] < chunk_priority) | \
               (priority_map[row_start:row_end, col_start:col_end] == 0)

        valid_mask = mask & (chunk_slice > 0)

        full_raster[row_start:row_end, col_start:col_end][valid_mask] = chunk_slice[valid_mask]
        priority_map[row_start:row_end, col_start:col_end][valid_mask] = chunk_priority[valid_mask]

    building_pixels = (full_raster > 0).sum()
    nodata_pixels = (full_raster == -999).sum()
    total_pixels = full_raster.size

    logger.info("Stitched raster: %d building pixels (%.1f%%), %d NoData (%.1f%%)",
                building_pixels, building_pixels / total_pixels * 100,
                nodata_pixels, nodata_pixels / total_pixels * 100)

    return full_raster

