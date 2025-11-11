"""Chunk processor for parallel processing of large districts."""

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
    Calculate optimal number of chunks based on context.

    Strategy:
    - If MPI is used (size > 1): Use MPI-aware calculation
    - Otherwise: Use config value or building-based heuristic
    - Ensure each chunk has at least 100 buildings
    - Cap at config_threads * 2 to avoid excessive overhead

    Args:
        num_buildings: Number of buildings in the district
        mpi_size: Total number of MPI processes
        config_chunks: Configured chunk count from config/env
        config_threads: Number of parallel threads

    Returns:
        Optimal number of chunks
    """
    # Minimum buildings per chunk (avoid too small chunks)
    min_buildings_per_chunk = 100
    max_chunks_from_buildings = max(1, num_buildings // min_buildings_per_chunk)

    # If MPI is used, leverage worker count
    if mpi_size > 1:
        # Use (MPI workers - 1) * threads as base
        # -1 because one process is master
        mpi_workers = mpi_size - 1
        chunks_from_mpi = max(4, mpi_workers * 2)  # At least 4 chunks

        logger.info(
            "MPI mode: %d processes (%d workers) → suggesting %d chunks",
            mpi_size, mpi_workers, chunks_from_mpi
        )

        # Use MPI-based calculation, but respect building count limit
        num_chunks = min(chunks_from_mpi, max_chunks_from_buildings)
    else:
        # Non-MPI mode: use config value or heuristic
        # Cap at threads * 2 for good task queue depth
        max_chunks = config_threads * 2

        # Use config value, but adjust based on building count
        num_chunks = min(config_chunks, max_chunks, max_chunks_from_buildings)

    # Final bounds
    num_chunks = max(2, min(num_chunks, 32))  # Between 2 and 32

    logger.info(
        "Chunk calculation: %d buildings, MPI=%s → %d chunks (%.0f buildings/chunk)",
        num_buildings,
        f"{mpi_size} processes" if mpi_size > 1 else "disabled",
        num_chunks,
        num_buildings / num_chunks
    )

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
    Process a single chunk (similar to process_district but for a chunk).

    Args:
        chunk_id: Identifier for this chunk
        chunk_geom: Shapely geometry of the chunk
        chunk_buildings: GeoDataFrame of buildings in this chunk
        config: Configuration object
        rasterizer: Rasterizer instance
        voronoi_generator: VoronoiGenerator instance
        district_attrs: District attributes to propagate

    Returns:
        Tuple of (voronoi_gdf, voronoi_raster, transform, bounds, buildings) or None if failed
    """
    logger.info("Processing chunk %d with %d buildings", chunk_id, len(chunk_buildings))

    try:
        # Rasterize buildings in this chunk
        raster, transform, bounds = rasterizer.rasterize_buildings(
            chunk_geom, chunk_buildings
        )

        if raster.max() == 0:
            logger.warning("Empty raster for chunk %d, skipping", chunk_id)
            return None

        logger.debug("Chunk %d rasterized: %dx%d pixels", chunk_id, raster.shape[0], raster.shape[1])

        # Generate Voronoi diagram
        district_mask = rasterizer.rasterize_district_mask(
            chunk_geom, transform, raster.shape
        )

        # Generate Voronoi polygons (disable visualization for chunks to avoid clutter)
        voronoi_gdf, voronoi_raster = voronoi_generator.generate_voronoi_polygons(
            building_raster=raster,
            district_mask=district_mask,
            transform=transform,
            crs="EPSG:32650",
            district_attrs=district_attrs,
            visualize=False,  # Disable visualization for chunks
            viz_interval=config.viz_interval,
            debug_mode=False  # Disable debug for chunks
        )

        if len(voronoi_gdf) == 0:
            logger.warning("No Voronoi polygons generated for chunk %d", chunk_id)
            return None

        logger.info("Chunk %d completed: %d Voronoi polygons, %.2f m² total area",
                   chunk_id, len(voronoi_gdf), voronoi_gdf['area'].sum())

        return voronoi_gdf, voronoi_raster, transform, bounds, chunk_buildings

    except Exception as e:
        logger.error("Error processing chunk %d: %s", chunk_id, e, exc_info=True)
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
    Process a large district using chunked parallel processing.

    Args:
        config: Configuration object
        reader: ShapefileReader instance
        rasterizer: Rasterizer instance
        district_row: District row from GeoDataFrame
        idx: District index
        voronoi_generator: VoronoiGenerator instance
        mpi_size: Total number of MPI processes (default: 1 for non-MPI mode)

    Returns:
        True if processing succeeded, False otherwise
    """
    # Try both uppercase and lowercase FID field names
    district_id = district_row.get("FID")
    district_geom = district_row.geometry

    logger.info("\n%s", "="*80)
    logger.info("Processing large district %s using CHUNKED mode", district_id)
    logger.info("%s", "="*80)

    # Get buildings in this district
    buildings = reader.get_buildings_in_district(district_geom)

    if len(buildings) == 0:
        logger.warning("No buildings found in district %s, skipping", district_id)
        return True  # Return True as this is a valid case, not an error

    logger.info("Found %d buildings in district", len(buildings))

    # Prepare district attributes
    district_attrs = {'district_id': district_id}
    for col in district_row.index:
        if col != 'geometry' and col != 'FID':
            district_attrs[col] = district_row[col]

    # Dynamically determine number of chunks based on context
    num_chunks = _calculate_optimal_chunks(
        num_buildings=len(buildings),
        mpi_size=mpi_size,
        config_chunks=config.num_chunks,
        config_threads=config.chunk_threads
    )

    # Split district into chunks
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

    logger.info("Created %d chunks, processing in parallel with %d threads",
                len(chunks), config.chunk_threads)

    # Process chunks in parallel
    chunk_results = []

    with ThreadPoolExecutor(max_workers=config.chunk_threads) as executor:
        # Submit all chunks for processing
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

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                if result is not None:
                    chunk_results.append((chunk_idx, result))
                    logger.info("Chunk %d completed successfully", chunk_idx)
                else:
                    logger.warning("Chunk %d returned no result", chunk_idx)
            except Exception as exc:
                logger.error("Chunk %d generated exception: %s", chunk_idx, exc, exc_info=True)

    if len(chunk_results) == 0:
        logger.error("No chunks processed successfully for district %s", district_id)
        return False

    logger.info("Completed processing %d/%d chunks, now merging results",
                len(chunk_results), len(chunks))

    # Sort results by chunk_idx
    chunk_results.sort(key=lambda x: x[0])

    # Extract components for merging
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

    # Merge Voronoi polygons with building-aware logic
    logger.info("Merging Voronoi polygons from chunks...")
    merged_voronoi_gdf = merge_voronoi_chunks(
        chunk_voronoi_gdfs,
        overlap_width=config.chunk_overlap
    )

    if len(merged_voronoi_gdf) == 0:
        logger.error("Failed to merge Voronoi polygons for district %s", district_id)
        return False

    # Save merged Voronoi polygons
    output_path = config.voronoi_dir / f"district_{district_id}_voronoi.shp"
    config.voronoi_dir.mkdir(parents=True, exist_ok=True)
    merged_voronoi_gdf.to_file(output_path)
    logger.info("Merged Voronoi polygons saved to %s (%d features, %.2f m² total)",
               output_path, len(merged_voronoi_gdf), merged_voronoi_gdf['area'].sum())

    # Merge voronoi rasters
    logger.info("Stitching Voronoi rasters from chunks...")

    # Calculate full raster dimensions and transform
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

    # Save merged Voronoi raster
    voronoi_raster_path = config.voronoi_dir / f"district_{district_id}_voronoi_raster.tif"
    rasterizer.save_raster_as_tif(
        merged_voronoi_raster.astype('int32'),
        full_transform,
        voronoi_raster_path,
        nodata=-999
    )
    logger.info("Merged Voronoi raster saved to %s", voronoi_raster_path)

    # Compute adjacency matrices for each chunk first
    logger.info("Computing adjacency matrices for chunks...")
    chunk_adjacencies = []

    for chunk_idx, (voronoi_gdf, _, _, _, chunk_buildings) in chunk_results:
        try:
            adjacency_matrix = create_adjacency_matrix(voronoi_gdf, chunk_buildings)
            chunk_adjacencies.append(adjacency_matrix)
        except Exception as e:
            logger.warning("Failed to compute adjacency for chunk %d: %s", chunk_idx, e)

    # Merge adjacency matrices
    logger.info("Merging and re-validating adjacency matrices...")
    merged_adjacency = merge_adjacency_matrices(
        chunk_adjacencies,
        merged_voronoi_gdf,
        buildings
    )

    # Save merged adjacency matrix
    adjacency_path = config.voronoi_dir / f"district_{district_id}_adjacency.pkl"
    merged_adjacency.to_pickle(adjacency_path)
    logger.info("Merged adjacency matrix saved to %s (shape: %s)",
               adjacency_path, merged_adjacency.shape)

    # Export CSV for debugging if requested
    if config.debug_adjacency:
        csv_path = config.voronoi_dir / f"district_{district_id}_adjacency.csv"
        merged_adjacency.to_csv(csv_path)
        logger.info("Adjacency matrix CSV exported to %s for debugging", csv_path)

    logger.info("%s", "="*80)
    logger.info("Chunked processing completed successfully for district %s", district_id)
    logger.info("%s\n", "="*80)

    return True

