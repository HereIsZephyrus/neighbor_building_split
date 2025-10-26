def process_district(config, reader, logger, rasterizer, district_row, idx):
    """Process a district."""
    district_id = district_row.get("id", idx)
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

    # Export raster if not in training mode
    if config.generate_raw_raster:
        raster_path = config.image_dir / f"district_{district_id}_raster.tif"
        rasterizer.save_raster_as_tif(raster, transform, raster_path)
        logger.info("Raster saved to %s", raster_path)
