"""District processing orchestrator."""

from .utils import get_logger, create_adjacency_matrix
from .chunker import estimate_raster_shape, calculate_required_chunks
from .chunk_processor import process_district_chunked

logger = get_logger(__name__)

def process_district(config, reader, rasterizer, district_row, idx, voronoi_generator=None, mpi_size=1):
    """
    Process a single district to generate Voronoi diagrams or rasters.
    
    Automatically switches between standard and chunked processing based on district size.
    """
    district_id = district_row.get("FID", district_row.get("fid", idx))
    district_geom = district_row.geometry

    # Skip if outputs already exist
    if config.generate_voronoi_diagram:
        voronoi_shp = config.voronoi_dir / f"district_{district_id}_voronoi.shp"
        adjacency_pkl = config.voronoi_dir / f"district_{district_id}_adjacency.pkl"
        if voronoi_shp.exists() and adjacency_pkl.exists():
            logger.info("District %s already processed, skipping", district_id)
            return
    elif config.generate_raw_raster:
        raster_path = config.image_dir / f"district_{district_id}_raster.tif"
        if raster_path.exists():
            logger.info("District %s already processed, skipping", district_id)
            return

    # Use chunked processing for large districts
    if config.enable_large_district_mode and config.generate_voronoi_diagram:
        estimated_shape = estimate_raster_shape(
            district_geom,
            resolution=rasterizer.resolution,
            buffer=10.0
        )

        needs_chunking = calculate_required_chunks(estimated_shape, config.max_chunk_size)

        if needs_chunking:
            logger.info("District %s size: %dx%d pixels, using chunked processing",
                       district_id, estimated_shape[0], estimated_shape[1])

            success = process_district_chunked(
                config, reader, rasterizer, district_row, idx, voronoi_generator, mpi_size
            )

            if not success:
                logger.warning("Chunked processing failed for district %s", district_id)

            return

    logger.info("\nProcessing district %s (area: %.2f m²)", district_id, 
               district_row.get('area', district_geom.area))

    buildings = reader.get_buildings_in_district(district_geom)

    if len(buildings) == 0:
        logger.warning("No buildings found in district %s", district_id)
        return

    logger.info("Found %d buildings", len(buildings))

    logger.debug("Rasterizing buildings...")
    raster, transform, _ = rasterizer.rasterize_buildings(
        district_geom, buildings
    )

    if raster.max() == 0:
        logger.warning("Empty raster for district %s", district_id)
        return

    if config.generate_raw_raster:
        raster_path = config.image_dir / f"district_{district_id}_raster.tif"
        rasterizer.save_raster_as_tif(raster, transform, raster_path)
        logger.info("Raster saved to %s", raster_path)

    if config.generate_voronoi_diagram and voronoi_generator is not None:
        logger.debug("Generating Voronoi polygons...")

        district_mask = rasterizer.rasterize_district_mask(
            district_geom, transform, raster.shape
        )

        district_attrs = {'district_id': district_id}
        for col in district_row.index:
            if col != 'geometry' and col != 'FID':
                district_attrs[col] = district_row[col]
        try:
            voronoi_gdf, voronoi_raster = voronoi_generator.generate_voronoi_polygons(
                building_raster=raster,
                district_mask=district_mask,
                transform=transform,
                crs="EPSG:32650",
                district_attrs=district_attrs,
                visualize=config.visualize_voronoi,
                viz_interval=config.viz_interval,
                debug_mode=config.debug_voronoi
            )

            if len(voronoi_gdf) > 0:
                output_path = config.voronoi_dir / f"district_{district_id}_voronoi.shp"
                config.voronoi_dir.mkdir(parents=True, exist_ok=True)
                voronoi_gdf.to_file(output_path)
                logger.info("Saved %d Voronoi polygons (%.2f m²)",
                           len(voronoi_gdf), voronoi_gdf['area'].sum())

                voronoi_raster_path = config.voronoi_dir / f"district_{district_id}_voronoi_raster.tif"
                rasterizer.save_raster_as_tif(
                    voronoi_raster.astype('int32'),
                    transform,
                    voronoi_raster_path,
                    nodata=-999
                )
                logger.debug("Voronoi raster saved to %s", voronoi_raster_path)

                logger.debug("Computing adjacency matrix...")
                adjacency_matrix = create_adjacency_matrix(voronoi_gdf, buildings)
                adjacency_path = config.voronoi_dir / f"district_{district_id}_adjacency.pkl"
                adjacency_matrix.to_pickle(adjacency_path)
                logger.info("Adjacency matrix saved (shape: %s)", adjacency_matrix.shape)

                if config.debug_adjacency:
                    csv_path = config.voronoi_dir / f"district_{district_id}_adjacency.csv"
                    adjacency_matrix.to_csv(csv_path)
                    logger.debug("Adjacency CSV exported for debugging")
            else:
                logger.warning("No Voronoi polygons generated for district %s", district_id)

        except Exception as e:
            logger.error("Failed to generate Voronoi for district %s: %s",
                        district_id, e, exc_info=True)
