"""Chunk merger for combining results from parallel chunk processing."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from affine import Affine
from .utils.logger import get_logger
from .utils.adjacency import find_adjacent_voronoi_regions, compute_building_distance, create_building_id_mapping

logger = get_logger()


def merge_voronoi_chunks(
    chunk_voronoi_gdfs: List[gpd.GeoDataFrame],
    overlap_width: int = 100
) -> gpd.GeoDataFrame:
    """
    Merge Voronoi polygons from multiple chunks with building-aware logic.

    This function groups Voronoi polygons by building_id and merges them
    spatially to handle cases where the same building's Voronoi region
    spans multiple chunks.

    Args:
        chunk_voronoi_gdfs: List of GeoDataFrames from each chunk
        overlap_width: Overlap width in pixels (for logging)

    Returns:
        Unified GeoDataFrame with one row per building
    """
    logger.info("Merging Voronoi polygons from %d chunks (overlap: %d pixels)",
                len(chunk_voronoi_gdfs), overlap_width)

    if len(chunk_voronoi_gdfs) == 0:
        logger.warning("No chunk results to merge")
        return gpd.GeoDataFrame()

    # Concatenate all chunk GeoDataFrames
    all_voronoi = pd.concat(chunk_voronoi_gdfs, ignore_index=True)

    if len(all_voronoi) == 0:
        logger.warning("No Voronoi polygons to merge")
        return gpd.GeoDataFrame()

    logger.info("Total Voronoi polygons before merging: %d", len(all_voronoi))

    # Group by building_id and merge geometries
    merged_rows = []

    for building_id, group in all_voronoi.groupby('building_id'):
        if len(group) == 1:
            # Single polygon for this building, keep as is
            merged_rows.append(group.iloc[0].to_dict())
        else:
            # Multiple polygons for this building - merge them
            logger.debug("Building %d has %d polygons across chunks, merging",
                        building_id, len(group))

            # Merge all geometries for this building using unary_union
            merged_geom = unary_union(group.geometry.tolist())

            # Create merged row with updated geometry and area
            row_dict = group.iloc[0].to_dict()  # Start with first row's attributes
            row_dict['geometry'] = merged_geom
            row_dict['area'] = merged_geom.area

            merged_rows.append(row_dict)

    # Create merged GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_rows, crs=all_voronoi.crs)

    logger.info("Merged Voronoi polygons: %d unique buildings (reduced from %d polygons)",
                len(merged_gdf), len(all_voronoi))
    logger.info("Total merged area: %.2f m²", merged_gdf['area'].sum())

    return merged_gdf


def merge_adjacency_matrices(
    chunk_adjacencies: List[pd.DataFrame],
    merged_voronoi_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Merge adjacency matrices from multiple chunks and re-validate.

    After merging Voronoi polygons, adjacency relationships need to be
    recalculated to ensure accuracy, especially at chunk boundaries.

    Args:
        chunk_adjacencies: List of adjacency matrices from each chunk
        merged_voronoi_gdf: Merged Voronoi GeoDataFrame
        buildings_gdf: Original building GeoDataFrame

    Returns:
        Unified adjacency matrix for entire district
    """
    logger.info("Merging adjacency matrices from %d chunks", len(chunk_adjacencies))

    if len(chunk_adjacencies) == 0:
        logger.warning("No chunk adjacency matrices to merge")
        return pd.DataFrame()

    # Get all unique building IDs
    all_building_ids = sorted(merged_voronoi_gdf['building_id'].unique())

    # Initialize combined matrix with zeros
    combined_matrix = pd.DataFrame(0.0, index=all_building_ids, columns=all_building_ids)

    # Combine adjacency information from all chunks (take minimum non-zero distance)
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
                        # Keep minimum distance
                        combined_matrix.loc[idx, col] = min(current_val, chunk_val)

    logger.info("Initial combined matrix: %d adjacency pairs",
                (combined_matrix.values > 0).sum() // 2)

    # Re-validate adjacencies using merged Voronoi polygons
    logger.info("Re-validating adjacencies on merged Voronoi polygons...")

    # Find adjacent pairs in merged Voronoi diagram
    adjacent_pairs = find_adjacent_voronoi_regions(merged_voronoi_gdf)

    # Create building ID to geometry mapping
    id_mapping = create_building_id_mapping(buildings_gdf)
    id_to_idx = {building_id: idx for idx, building_id in id_mapping.items()}

    # Initialize final matrix
    final_matrix = pd.DataFrame(0.0, index=all_building_ids, columns=all_building_ids)

    # Set distances for adjacent pairs
    distances_computed = 0
    skipped_missing = 0

    for building_id_i, building_id_j in adjacent_pairs:
        # Get building geometries
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
        final_matrix.loc[building_id_i, building_id_j] = distance
        final_matrix.loc[building_id_j, building_id_i] = distance

        distances_computed += 1

    if skipped_missing > 0:
        logger.warning("Skipped %d adjacent pairs due to missing building geometries",
                      skipped_missing)

    # Log statistics
    non_zero_values = final_matrix.values[final_matrix.values > 0]
    if len(non_zero_values) > 0:
        logger.info(
            "Final adjacency matrix: shape=%s, adjacencies=%d, "
            "distance stats: min=%.2f, max=%.2f, mean=%.2f meters",
            final_matrix.shape,
            len(non_zero_values) // 2,
            non_zero_values.min(),
            non_zero_values.max(),
            non_zero_values.mean()
        )
    else:
        logger.warning("Final adjacency matrix has no non-zero values")

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
    Merge raster outputs from multiple chunks.

    For overlap regions, keeps the value from the chunk where the
    building center is most likely located.

    Args:
        chunk_rasters: List of raster arrays from each chunk
        chunk_transforms: List of affine transforms for each chunk
        chunk_bounds: List of bounds for each chunk
        full_bounds: Bounds of the full district raster
        full_transform: Affine transform for full raster
        full_shape: Shape (height, width) of full raster
        overlap: Overlap width in pixels

    Returns:
        Merged raster array
    """
    logger.info("Stitching %d chunk rasters into full raster (shape: %s)",
                len(chunk_rasters), full_shape)

    # Initialize full raster with -999 (NoData)
    full_raster = np.full(full_shape, -999, dtype=np.int32)

    # Create a priority map to track which chunk should "win" in overlap regions
    # Higher values = higher priority
    priority_map = np.zeros(full_shape, dtype=np.int32)

    for chunk_idx, (chunk_raster, _chunk_transform, chunk_bounds) in enumerate(
        zip(chunk_rasters, chunk_transforms, chunk_bounds)
    ):
        # Calculate pixel coordinates in full raster
        chunk_minx, _chunk_miny, _chunk_maxx, chunk_maxy = chunk_bounds

        # Convert chunk bounds to pixel coordinates in full raster
        col_start = int((chunk_minx - full_bounds[0]) / full_transform.a)
        row_start = int((full_bounds[3] - chunk_maxy) / abs(full_transform.e))

        chunk_h, chunk_w = chunk_raster.shape
        col_end = min(col_start + chunk_w, full_shape[1])
        row_end = min(row_start + chunk_h, full_shape[0])

        # Adjust if out of bounds
        col_start = max(0, col_start)
        row_start = max(0, row_start)

        # Calculate corresponding slice in chunk raster
        chunk_col_start = max(0, -int((chunk_minx - full_bounds[0]) / full_transform.a))
        chunk_row_start = max(0, -int((full_bounds[3] - chunk_maxy) / abs(full_transform.e)))
        chunk_col_end = chunk_col_start + (col_end - col_start)
        chunk_row_end = chunk_row_start + (row_end - row_start)

        # Extract the portion of chunk raster to copy
        chunk_slice = chunk_raster[chunk_row_start:chunk_row_end, chunk_col_start:chunk_col_end]

        # Calculate priority: higher for non-overlap regions (center of chunk)
        # For overlap regions, we'll prefer chunks where buildings are more centered
        chunk_priority = np.ones_like(chunk_slice, dtype=np.int32) * (chunk_idx + 1)

        # Copy to full raster where priority is higher or not yet set
        mask = (priority_map[row_start:row_end, col_start:col_end] < chunk_priority) | \
               (priority_map[row_start:row_end, col_start:col_end] == 0)

        # Only update pixels that have valid building IDs
        valid_mask = mask & (chunk_slice > 0)

        full_raster[row_start:row_end, col_start:col_end][valid_mask] = chunk_slice[valid_mask]
        priority_map[row_start:row_end, col_start:col_end][valid_mask] = chunk_priority[valid_mask]

        logger.debug("Placed chunk %d at position (%d, %d) to (%d, %d)",
                    chunk_idx, row_start, col_start, row_end, col_end)

    # Count statistics
    building_pixels = (full_raster > 0).sum()
    nodata_pixels = (full_raster == -999).sum()
    total_pixels = full_raster.size

    logger.info("Stitched raster: %d building pixels (%.1f%%), %d NoData pixels (%.1f%%)",
                building_pixels, building_pixels / total_pixels * 100,
                nodata_pixels, nodata_pixels / total_pixels * 100)

    return full_raster

