"""Main script for building pattern segmentation."""

import sys
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from .utils import Config, setup_logger
from .reader import ShapefileReader
from .converter import Rasterizer, VoronoiGenerator
from .processor import process_district

# Try to import MPI, but don't fail if not available
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Building Pattern Segmentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Generate raster for training
          python -m src --generate-raster-for-training --district-path /path/to/district.shp

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
    mode_group.add_argument(
        "--generate-raster-for-training",
        action="store_true",
        help="Generate raster for training"
    )

    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument(
        "--district-path",
        type=str,
        help="Path to district shapefile (overrides DISTRICT in .env)"
    )

    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize Voronoi dilation process with OpenCV (only with --generate-voronoi-diagram)"
    )
    viz_group.add_argument(
        "--viz-interval",
        type=int,
        default=1,
        help="Show visualization every N iterations (default: 1, only with --visualize)"
    )
    viz_group.add_argument(
        "--debug-voronoi",
        action="store_true",
        help="Debug mode: press SPACE to step through each iteration (only with --visualize)"
    )
    viz_group.add_argument(
        "--debug-adjacency",
        action="store_true",
        help="Debug mode: export adjacency matrix as CSV in addition to pickle format"
    )

    # Parallel processing options
    parallel_group = parser.add_argument_group('Parallel Processing Options')
    parallel_group.add_argument(
        "--use-mpi",
        action="store_true",
        help="Enable MPI parallel processing (requires mpirun/mpiexec and mpi4py)"
    )

    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Determine execution mode
    generate_voronoi_diagram = args.generate_voronoi_diagram
    generate_raw_raster = args.generate_raster_for_training

    # Initialize MPI if requested
    use_mpi = args.use_mpi
    comm = None
    rank = 0
    size = 1

    if use_mpi:
        if not MPI_AVAILABLE:
            print("ERROR: MPI requested but mpi4py is not available.")
            print("Please install mpi4py: pip install mpi4py")
            sys.exit(1)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    # Load configuration with optional parameter overrides
    try:
        config = Config(
            district_path=args.district_path,
            generate_raw_raster=generate_raw_raster,
            generate_voronoi_diagram=generate_voronoi_diagram,
            visualize_voronoi=args.visualize,
            viz_interval=args.viz_interval,
            debug_voronoi=args.debug_voronoi,
            debug_adjacency=args.debug_adjacency
        )
    except ValueError as e:
        if rank == 0:
            print(f"Configuration error: {e}")
            print("Please provide required paths via arguments or .env file.")
        sys.exit(1)

    # Setup logger (each rank gets its own log file if using MPI)
    if use_mpi:
        log_suffix = f"_rank{rank}" if size > 1 else ""
    else:
        log_suffix = ""

    log_file = config.output_dir / f"voronoi_diagram_{datetime.now().strftime('%Y%m%d_%H')}{log_suffix}.log" if generate_voronoi_diagram else config.output_dir / f"raster_generation_{datetime.now().strftime('%Y%m%d_%H')}{log_suffix}.log"
    mode_name = "Voronoi Diagram Generation" if generate_voronoi_diagram else "Raster Generation"

    logger = setup_logger(log_file=log_file, level = logging.INFO)

    if rank == 0:
        logger.info("=" * 80)
        logger.info(mode_name)
        if use_mpi:
            logger.info("MPI Enabled: %d processes", size)
        logger.info("=" * 80)

    # Initialize components
    logger.info("Initializing components...")
    reader = ShapefileReader(config.district_path, config.building_path)
    rasterizer = Rasterizer(resolution=1.0)

    # Initialize Voronoi generator if needed
    voronoi_generator = None
    if generate_voronoi_diagram:
        voronoi_generator = VoronoiGenerator(simplify_tolerance=0.1)
        logger.info("Voronoi generator initialized")

    # Load data
    logger.info("Loading shapefiles...")
    districts = reader.load_districts()
    reader.load_buildings()

    # Sort districts by area (ascending) to process small ones first
    if 'geometry' in districts.columns:
        districts['area'] = districts.geometry.area
        districts = districts.sort_values('area', ascending=True).reset_index(drop=True)
        if rank == 0:
            logger.info("Sorted %d districts by area (min: %.2f m², max: %.2f m²)", 
                       len(districts), districts['area'].min(), districts['area'].max())

    if rank == 0:
        logger.info("Processing %d districts...", len(districts))

    # Distribute work among MPI processes if enabled
    if use_mpi and size > 1:
        # Divide districts among processes
        # Each process gets approximately len(districts) / size districts
        districts_list = list(districts.iterrows())
        my_districts = districts_list[rank::size]  # Interleaved distribution

        logger.info("Rank %d processing %d districts (indices: %s)", 
                   rank, len(my_districts), 
                   [idx for idx, _ in my_districts[:5]] + (['...'] if len(my_districts) > 5 else []))

        # Process assigned districts (no tqdm in parallel mode)
        completed = 0
        for idx, district_row in my_districts:
            try:
                process_district(
                    config, reader, rasterizer, district_row, idx,
                    voronoi_generator=voronoi_generator
                )
                completed += 1
                # Log progress periodically
                if completed % 10 == 0 or completed == len(my_districts):
                    logger.info("Rank %d: Completed %d/%d districts (%.1f%%)",
                               rank, completed, len(my_districts),
                               completed / len(my_districts) * 100)
            except Exception as exc:
                logger.error("Error processing district %s: %s",
                            idx, exc, exc_info=True)
                continue

        logger.info("Rank %d finished processing all %d districts", rank, completed)

        # Synchronize all processes before finishing
        if comm is not None:
            comm.Barrier()
            if rank == 0:
                logger.info("All MPI processes completed")
    else:
        # Sequential processing (no MPI or single process)
        for idx, district_row in tqdm(
            districts.iterrows(), total=len(districts), desc="Processing districts"
        ):
            try:
                process_district(
                    config, reader, rasterizer, district_row, idx,
                    voronoi_generator=voronoi_generator
                )
            except Exception as exc:
                logger.error("Error processing district %s: %s",
                            idx, exc, exc_info=True)
                continue

if __name__ == "__main__":
    main()
