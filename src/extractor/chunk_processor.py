"""Parallel chunk processing for large districts."""

from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import geopandas as gpd
from affine import Affine
from .utils.logger import get_logger
from .utils.adjacency import create_adjacency_matrix
from .chunker import split_district_adaptive
from .chunk_merger import merge_voronoi_chunks, merge_adjacency_matrices, stitch_rasters

logger = get_logger(__name__)


def _calculate_optimal_chunks(
    num_buildings: int,
    mpi_size: int,
    config_chunks: int,
    config_threads: int
) -> int:
    """
    Calculate optimal chunk count balancing parallelism and overhead.

    Ensures at least 100 buildings per chunk, caps at 32 chunks total.
    MPI-aware: scales with worker count when MPI is enabled.
    """
    min_buildings_per_chunk = 100
    max_chunks_from_buildings = max(1, num_buildings // min_buildings_per_chunk)

    if mpi_size > 1:
        mpi_workers = mpi_size - 1
        chunks_from_mpi = max(4, mpi_workers * 2)
        logger.debug("MPI mode: %d workers → %d chunks", mpi_workers, chunks_from_mpi)
        num_chunks = min(chunks_from_mpi, max_chunks_from_buildings)
    else:
        max_chunks = config_threads * 2
        num_chunks = min(config_chunks, max_chunks, max_chunks_from_buildings)

    num_chunks = max(2, min(num_chunks, 32))

    logger.info("Using %d chunks (%.0f buildings/chunk)", num_chunks, num_buildings / num_chunks)

    return num_chunks


def process_chunk(
    chunk_id: int,
    chunk_geom,
    chunk_buildings: gpd.GeoDataFrame,
    config,
    rasterizer,
    voronoi_generator,
    district_attrs: Dict[str, Any]
) -> Optional[Tuple[gpd.GeoDataFrame, np.ndarray, Affine, Tuple[float, float, float, float], gpd.GeoDataFrame]]:
    """
    Process a single spatial chunk to generate Voronoi diagram.

    Returns (voronoi_gdf, voronoi_raster, transform, bounds, buildings) or None on failure.
    """
    logger.debug("Processing chunk %d with %d buildings", chunk_id, len(chunk_buildings))

    try:
        raster, transform, bounds = rasterizer.rasterize_buildings(
            chunk_geom, chunk_buildings
        )

        if raster.max() == 0:
            logger.warning("Chunk %d: empty raster", chunk_id)
            return None

        district_mask = rasterizer.rasterize_district_mask(
            chunk_geom, transform, raster.shape
        )

        voronoi_gdf, voronoi_raster = voronoi_generator.generate_voronoi_polygons(
            building_raster=raster,
            district_mask=district_mask,
            transform=transform,
            crs="EPSG:32650",
            district_attrs=district_attrs,
            visualize=False,
            viz_interval=config.viz_interval,
            debug_mode=False
        )

        if len(voronoi_gdf) == 0:
            logger.warning("Chunk %d: no Voronoi polygons", chunk_id)
            return None

        logger.debug("Chunk %d: %d polygons, %.2f m²",
                   chunk_id, len(voronoi_gdf), voronoi_gdf['area'].sum())

        return voronoi_gdf, voronoi_raster, transform, bounds, chunk_buildings

    except Exception as e:
        logger.error("Chunk %d failed: %s", chunk_id, e, exc_info=True)
        return None


def process_district_chunked(
    config,
    reader,
    rasterizer,
    district_row,
    idx,
    voronoi_generator,
    mpi_size: int = 1
) -> bool:
    """
    Process large district using spatial partitioning and parallel workers.

    Returns True on success, False on failure.
    """
    district_id = district_row.get("FID")
    district_geom = district_row.geometry

    logger.info("\n%s", "="*80)
    logger.info("Chunked processing: district %s", district_id)
    logger.info("%s", "="*80)

    buildings = reader.get_buildings_in_district(district_geom)

    if len(buildings) == 0:
        logger.warning("No buildings in district %s", district_id)
        return True

    logger.info("Found %d buildings", len(buildings))

    district_attrs = {'district_id': district_id}
    for col in district_row.index:
        if col != 'geometry' and col != 'FID':
            district_attrs[col] = district_row[col]

    num_chunks = _calculate_optimal_chunks(
        num_buildings=len(buildings),
        mpi_size=mpi_size,
        config_chunks=config.num_chunks,
        config_threads=config.chunk_threads
    )

    chunks = split_district_adaptive(
        district_geom,
        buildings,
        num_chunks=num_chunks,
        overlap=config.chunk_overlap,
        resolution=rasterizer.resolution
    )

    if len(chunks) == 0:
        logger.error("Failed to create chunks for district %s", district_id)
        return False

    logger.info("Processing %d chunks with %d threads", len(chunks), config.chunk_threads)

    chunk_results = []

    with ThreadPoolExecutor(max_workers=config.chunk_threads) as executor:
        future_to_chunk = {}
        for chunk_idx, (chunk_geom, chunk_buildings) in enumerate(chunks):
            future = executor.submit(
                process_chunk,
                chunk_idx,
                chunk_geom,
                chunk_buildings,
                config,
                rasterizer,
                voronoi_generator,
                district_attrs
            )
            future_to_chunk[future] = chunk_idx

        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                if result is not None:
                    chunk_results.append((chunk_idx, result))
                    logger.debug("Chunk %d completed", chunk_idx)
                else:
                    logger.warning("Chunk %d: no result", chunk_idx)
            except Exception as exc:
                logger.error("Chunk %d exception: %s", chunk_idx, exc, exc_info=True)

    if len(chunk_results) == 0:
        logger.error("No chunks succeeded for district %s", district_id)
        return False

    logger.info("Completed %d/%d chunks, merging...", len(chunk_results), len(chunks))

    chunk_results.sort(key=lambda x: x[0])

    chunk_voronoi_gdfs = []
    chunk_voronoi_rasters = []
    chunk_transforms = []
    chunk_bounds_list = []
    chunk_buildings_list = []

    for chunk_idx, (voronoi_gdf, voronoi_raster, transform, bounds, chunk_buildings) in chunk_results:
        chunk_voronoi_gdfs.append(voronoi_gdf)
        chunk_voronoi_rasters.append(voronoi_raster)
        chunk_transforms.append(transform)
        chunk_bounds_list.append(bounds)
        chunk_buildings_list.append(chunk_buildings)

    logger.debug("Merging Voronoi polygons...")
    merged_voronoi_gdf = merge_voronoi_chunks(
        chunk_voronoi_gdfs,
        overlap_width=config.chunk_overlap
    )

    if len(merged_voronoi_gdf) == 0:
        logger.error("Merge failed for district %s", district_id)
        return False

    output_path = config.voronoi_dir / f"district_{district_id}_voronoi.shp"
    config.voronoi_dir.mkdir(parents=True, exist_ok=True)
    merged_voronoi_gdf.to_file(output_path)
    logger.info("Saved %d merged polygons (%.2f m²)",
               len(merged_voronoi_gdf), merged_voronoi_gdf['area'].sum())

    logger.debug("Stitching rasters...")

    full_raster, full_transform, full_bounds = rasterizer.rasterize_buildings(
        district_geom, buildings
    )

    merged_voronoi_raster = stitch_rasters(
        chunk_voronoi_rasters,
        chunk_transforms,
        chunk_bounds_list,
        full_bounds,
        full_transform,
        full_raster.shape,
        overlap=config.chunk_overlap
    )

    voronoi_raster_path = config.voronoi_dir / f"district_{district_id}_voronoi_raster.tif"
    rasterizer.save_raster_as_tif(
        merged_voronoi_raster.astype('int32'),
        full_transform,
        voronoi_raster_path,
        nodata=-999
    )
    logger.debug("Raster saved to %s", voronoi_raster_path)

    logger.debug("Computing chunk adjacencies...")
    chunk_adjacencies = []

    for chunk_idx, (voronoi_gdf, _, _, _, chunk_buildings) in chunk_results:
        try:
            adjacency_matrix = create_adjacency_matrix(voronoi_gdf, chunk_buildings)
            chunk_adjacencies.append(adjacency_matrix)
        except Exception as e:
            logger.warning("Chunk %d adjacency failed: %s", chunk_idx, e)

    logger.debug("Merging adjacency matrices...")
    merged_adjacency = merge_adjacency_matrices(
        chunk_adjacencies,
        merged_voronoi_gdf,
        buildings
    )

    adjacency_path = config.voronoi_dir / f"district_{district_id}_adjacency.pkl"
    merged_adjacency.to_pickle(adjacency_path)
    logger.info("Adjacency matrix saved (shape: %s)", merged_adjacency.shape)

    if config.debug_adjacency:
        csv_path = config.voronoi_dir / f"district_{district_id}_adjacency.csv"
        merged_adjacency.to_csv(csv_path)
        logger.debug("Adjacency CSV exported for debugging")

    logger.info("%s", "="*80)
    logger.info("Chunked processing completed for district %s", district_id)
    logger.info("%s\n", "="*80)

    return True

