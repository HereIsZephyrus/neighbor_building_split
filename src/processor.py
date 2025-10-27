from .utils import get_logger

logger = get_logger()

def process_district(config, reader, rasterizer, district_row, idx, voronoi_generator=None):
    """Process a district."""
    district_id = district_row.get("FID", idx)
    district_geom = district_row.geometry
    logger.info("\nProcessing district %s", district_id)

    # Get buildings in this district
    buildings = reader.get_buildings_in_district(district_geom)

    if len(buildings) == 0:
        logger.warning("No buildings found in district %s, skipping", district_id)
        return

    logger.info("Found %d buildings in district", len(buildings))

    # Rasterize district and buildings
    logger.info("Rasterizing buildings...")
    raster, transform, _ = rasterizer.rasterize_buildings(
        district_geom, buildings
    )

    if raster.max() == 0:
        logger.warning("Empty raster for district %s, skipping", district_id)
        return

    # Export raster if in raster generation mode
    if config.generate_raw_raster:
        raster_path = config.image_dir / f"district_{district_id}_raster.tif"
        rasterizer.save_raster_as_tif(raster, transform, raster_path)
        logger.info("Raster saved to %s", raster_path)

    # Generate Voronoi diagram if in voronoi mode
    if config.generate_voronoi_diagram and voronoi_generator is not None:
        logger.info("Generating Voronoi boundaries...")

        # Create district mask
        district_mask = rasterizer.rasterize_district_mask(
            district_geom, transform, raster.shape
        )

        # Prepare district attributes
        district_attrs = {
            'district_id': district_id
        }
        # Copy other attributes from district row
        for col in district_row.index:
            if col != 'geometry' and col != 'FID':
                district_attrs[col] = district_row[col]

        # Generate Voronoi boundaries
        try:
            boundary_gdf, voronoi_raster = voronoi_generator.generate_voronoi_boundaries(
                building_raster=raster,
                district_mask=district_mask,
                transform=transform,
                crs="EPSG:32650",
                district_attrs=district_attrs,
                visualize=config.visualize_voronoi,
                viz_interval=config.viz_interval
            )

            if len(boundary_gdf) > 0:
                # Save boundary lines
                output_path = config.voronoi_dir / f"district_{district_id}_boundaries.shp"
                config.voronoi_dir.mkdir(parents=True, exist_ok=True)
                boundary_gdf.to_file(output_path)
                logger.info("Voronoi boundaries saved to %s (%d features, %.2f m total)",
                           output_path, len(boundary_gdf), boundary_gdf['length'].sum())

                # Optionally save Voronoi partition raster for debugging
                voronoi_raster_path = config.voronoi_dir / f"district_{district_id}_voronoi.tif"
                rasterizer.save_raster_as_tif(
                    voronoi_raster.astype('float32'),
                    transform,
                    voronoi_raster_path,
                    nodata=0
                )
                logger.info("Voronoi partition raster saved to %s", voronoi_raster_path)
            else:
                logger.warning("No boundary lines generated for district %s", district_id)

        except Exception as e:
            logger.error("Error generating Voronoi boundaries for district %s: %s",
                        district_id, e, exc_info=True)
