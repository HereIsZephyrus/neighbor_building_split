"""Main script for building pattern segmentation."""

import sys
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from .utils import Config, setup_logger
from .reader import ShapefileReader
from .converter import Rasterizer
from .processor import process_district

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Building Pattern Segmentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use CNN segmentation with custom district
  python -m src --use-cnn-segmentation --district-path /path/to/district.shp

  # Generate voronoi diagram
  python -m src --generate-voronoi-diagram --district-path /path/to/district.shp
        """
    )

    # Execution modes
    mode_group = parser.add_argument_group('Execution Modes')
    mode_group.add_argument(
        "--generate-voronoi-diagram",
        action="store_true",
        help="Generate voronoi diagram for the district"
    )

    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument(
        "--district-path",
        type=str,
        help="Path to district shapefile (overrides DISTRICT in .env)"
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Determine execution mode
    generate_voronoi_diagram = args.generate_voronoi_diagram
    generate_raw_raster = args.generate_raster_for_training

    # Load configuration with optional parameter overrides
    try:
        config = Config(
            district_path=args.district_path,
            generate_raw_raster=generate_raw_raster,
            generate_voronoi_diagram=generate_voronoi_diagram
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please provide required paths via arguments or .env file.")
        sys.exit(1)

    # Setup logger
    log_file = config.output_dir / f"voronoi_diagram_{datetime.now().strftime('%Y%m%d_%H')}.log" if generate_voronoi_diagram else config.output_dir / f"raster_generation_{datetime.now().strftime('%Y%m%d_%H')}.log"
    mode_name = "Voronoi Diagram Generation" if generate_voronoi_diagram else "Raster Generation"

    logger = setup_logger(log_file=log_file, level = logging.DEBUG)
    logger.info("=" * 80)
    logger.info(mode_name)
    logger.info("=" * 80)

    # Initialize components
    logger.info("Initializing components...")
    reader = ShapefileReader(config.district_path, config.building_path)
    rasterizer = Rasterizer(resolution=1.0)

    # Load data
    logger.info("Loading shapefiles...")
    districts = reader.load_districts()
    reader.load_buildings()

    logger.info("Processing %d districts...", len(districts))

    for idx, district_row in tqdm(
        districts.iterrows(), total=len(districts), desc="Processing districts"
    ):
        try:
            process_district(config, reader, logger, rasterizer, district_row, idx)
        except Exception as exc:
            logger.error("Error processing district %s: %s",
                        idx, exc, exc_info=True)
            continue

if __name__ == "__main__":
    main()
