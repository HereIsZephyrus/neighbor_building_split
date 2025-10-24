"""Main script for building pattern segmentation."""

import sys
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from .utils import Config, setup_logger
from .reader import ShapefileReader
from .converter import Rasterizer, Vectorizer
from .segmentation import FeatureExtractor, ClassicalSegmenter


def main():
    """Main execution function."""
    # Load configuration
    try:
        config = Config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please create a .env file with DISTRICT and BUILDING paths.")
        sys.exit(1)

    # Setup logger
    log_file = config.output_dir / f"segmentation_{datetime.now().strftime('%Y%m%d_%H')}.log"
    logger = setup_logger(log_file=log_file, level = logging.DEBUG)

    logger.info("=" * 80)
    logger.info("Building Pattern Segmentation")
    logger.info("=" * 80)
    logger.info("Configuration: %s", config.to_dict())

    try:
        # Initialize components
        logger.info("Initializing components...")
        reader = ShapefileReader(config.district_path, config.building_path)
        rasterizer = Rasterizer(resolution=1.0)
        vectorizer = Vectorizer(simplify_tolerance=1.0)
        feature_extractor = FeatureExtractor(
            use_cnn=config.use_cnn_features,
            cnn_model_name=config.cnn_model,
        )
        segmenter = ClassicalSegmenter(
            n_segments=config.n_segments,
            compactness=config.compactness,
            similarity_threshold=config.similarity_threshold,
        )

        # Load data
        logger.info("Loading shapefiles...")
        districts = reader.load_districts()
        reader.load_buildings()

        logger.info("Processing %d districts...", len(districts))

        # Process each district
        all_segments = []
        global_cluster_id = 1

        # Save raster as TIF for debugging
        image_dir = config.output_dir / "debug_rasters"
        image_dir.mkdir(exist_ok=True)

        for idx, district_row in tqdm(
            districts.iterrows(), total=len(districts), desc="Processing districts"
        ):
            try:
                district_id = district_row.get("id", idx)
                district_geom = district_row.geometry
                logger.info("\nProcessing district %s", district_id)

                # Get buildings in this district
                buildings = reader.get_buildings_in_district(district_geom)

                if len(buildings) == 0:
                    logger.warning("No buildings found in district %s, skipping", district_id)
                    continue

                logger.info("Found %d buildings in district", len(buildings))

                # Rasterize district and buildings
                logger.info("Rasterizing buildings...")
                raster, transform, _ = rasterizer.rasterize_buildings(
                    district_geom, buildings
                )

                if raster.max() == 0:
                    logger.warning("Empty raster for district %s, skipping", district_id)
                    continue

                raster_path = image_dir / f"district_{district_id}_raster.tif"
                rasterizer.save_raster_as_tif(raster, transform, raster_path)
                # Extract features
                logger.info("Extracting features...")
                features = feature_extractor.extract_features(raster)

                # Perform segmentation
                logger.info("Performing segmentation...")
                segmentation_result = segmenter.fit_predict(raster, features)

                # Vectorize results
                logger.info("Vectorizing segmentation results...")
                district_attrs = {
                    k: v for k, v in district_row.items() if k != "geometry"
                }
                segments_gdf = vectorizer.vectorize_segments(
                    segmentation_result,
                    transform,
                    crs="EPSG:32650",
                    district_attrs=district_attrs,
                )

                # Count buildings in each segment
                segments_gdf = vectorizer.count_buildings_in_segments(
                    segments_gdf, buildings
                )

                # Reassign cluster IDs to be globally continuous
                n_segments = len(segments_gdf)
                segments_gdf["cluster_id"] = range(
                    global_cluster_id, global_cluster_id + n_segments
                )
                global_cluster_id += n_segments

                all_segments.append(segments_gdf)
                logger.info(
                    "District %s: %d segments identified", district_id, n_segments
                )

            except Exception as exc:
                logger.error("Error processing district %s: %s",
                           district_id, exc, exc_info=True)
                continue

        # Merge all segments
        if not all_segments:
            logger.error("No segments generated from any district")
            sys.exit(1)

        logger.info("\nMerging all segments...")
        final_segments = pd.concat(all_segments, ignore_index=True)
        final_gdf = final_segments

        # Save output
        output_path = config.output_dir / "district_segments.shp"
        logger.info("Saving results to %s", output_path)
        final_gdf.to_file(output_path)

        # Summary
        logger.info("=" * 80)
        logger.info("Segmentation Complete")
        logger.info("=" * 80)
        logger.info("Total districts processed: %d", len(all_segments))
        logger.info("Total segments generated: %d", len(final_gdf))
        logger.info("Output saved to: %s", output_path)
        logger.info("Log saved to: %s", log_file)

    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
