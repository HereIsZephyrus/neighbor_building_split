"""Voronoi diagram generator using dilation method."""

from typing import Tuple, Optional
import numpy as np
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import linemerge
from affine import Affine
import cv2
from ..utils.logger import get_logger

logger = get_logger()


class VoronoiGenerator:
    """Class for generating Voronoi-like diagrams using dilation method."""

    def __init__(self, simplify_tolerance: float = 0.5):
        """
        Initialize Voronoi generator.

        Args:
            simplify_tolerance: Tolerance for line simplification in meters
        """
        self.simplify_tolerance = simplify_tolerance

    def generate_voronoi_from_raster(
        self,
        building_raster: np.ndarray,
        district_mask: np.ndarray,
        visualize: bool = False,
        viz_interval: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Voronoi partition from building raster using dilation method.

        This method:
        1. Erodes buildings to prevent false connections from rasterization
        2. Identifies connected components (buildings) in the eroded raster
        3. Uses dilation to restore original building shapes with labels
        4. Continues dilation to partition remaining space
        5. Respects district boundaries

        Args:
            building_raster: 2D array with building pixels > 0, background = 0, outside = -999
            district_mask: Binary mask (1 inside district, 0 outside)
            visualize: Whether to visualize the dilation process with OpenCV
            viz_interval: Show visualization every N iterations (default: 1)

        Returns:
            Tuple of (voronoi_partition, original_buildings_mask)
            - voronoi_partition: Array with each region labeled uniquely (0 for outside)
            - original_buildings_mask: Binary mask of original building pixels (before erosion)
        """
        logger.debug("Generating Voronoi diagram using dilation method")

        # Create binary mask for buildings (any positive value is a building)
        building_binary = (building_raster > 0).astype(np.uint8)
        original_buildings_mask = building_binary.copy()

        # Apply erosion to prevent false connections from rasterization artifacts
        # Use 3x3 structure for erosion
        erosion_structure = ndimage.generate_binary_structure(2, 1)  # 3x3 cross
        eroded_buildings = ndimage.binary_erosion(
            building_binary, structure=erosion_structure, iterations=1
        ).astype(np.uint8)

        logger.debug("Eroded buildings: %d pixels -> %d pixels",
                    building_binary.sum(), eroded_buildings.sum())

        # Identify connected components in eroded buildings as seeds
        # Use 4-connectivity to consider only direct neighbors
        structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
        labeled_buildings, num_features = ndimage.label(eroded_buildings, structure=structure)

        logger.info("Identified %d building connected components", num_features)

        if num_features == 0:
            logger.warning("No buildings found, returning empty Voronoi diagram")
            return np.zeros_like(building_raster, dtype=np.int32), original_buildings_mask

        # Initialize Voronoi partition with eroded building labels
        voronoi = labeled_buildings.copy()

        # First, dilate to restore original building shapes while keeping labels
        # This ensures buildings maintain their original shapes with proper labels
        voronoi = self._dilate_labels(
            voronoi, 
            district_mask, 
            max_iterations=40,
            visualize=visualize,
            viz_interval=viz_interval
        )

        logger.debug("Voronoi diagram generated with %d regions", num_features)

        return voronoi, original_buildings_mask

    def _generate_color_map(self, num_labels: int) -> np.ndarray:
        """
        Generate random colors for each label for visualization.
        
        Args:
            num_labels: Number of labels to generate colors for
            
        Returns:
            Color map array of shape (num_labels + 1, 3) with BGR colors
        """
        np.random.seed(42)  # For reproducibility
        colors = np.zeros((num_labels + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Black for unlabeled (background)
        
        # Generate random colors for each label
        for i in range(1, num_labels + 1):
            colors[i] = np.random.randint(0, 256, size=3, dtype=np.uint8)
        
        return colors
    
    def _visualize_labels(
        self,
        labeled_array: np.ndarray,
        color_map: np.ndarray,
        district_mask: np.ndarray,
        iteration: int,
        window_name: str = "Voronoi Dilation",
    ) -> None:
        """
        Visualize labeled array using color map.
        
        Args:
            labeled_array: Array with labels
            color_map: Color map for labels (BGR format)
            district_mask: Binary mask defining valid area
            iteration: Current iteration number
            window_name: Name of visualization window
        """
        # Create RGB visualization
        vis_img = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)
        
        # Apply colors based on labels
        for label in range(len(color_map)):
            mask = (labeled_array == label)
            vis_img[mask] = color_map[label]
        
        # Mark outside district area as dark gray
        outside_mask = (district_mask == 0)
        vis_img[outside_mask] = [40, 40, 40]
        
        # Add iteration counter text
        display_img = vis_img.copy()
        text = f"Iteration: {iteration}"
        cv2.putText(
            display_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Calculate remaining unlabeled pixels
        unlabeled = ((labeled_array == 0) & (district_mask == 1)).sum()
        progress_text = f"Unlabeled: {unlabeled} pixels"
        cv2.putText(
            display_img,
            progress_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Resize for better viewing if image is too small
        h, w = display_img.shape[:2]
        if h < 800 or w < 800:
            scale = max(800 / h, 800 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(window_name, display_img)
    
    def _dilate_labels(
        self,
        labeled_array: np.ndarray,
        district_mask: np.ndarray,
        max_iterations: int = 40,
        visualize: bool = False,
        viz_interval: int = 1,
        window_name: str = "Voronoi Dilation Progress",
    ) -> np.ndarray:
        """
        Dilate labeled regions to fill entire district using morphological dilation.

        Args:
            labeled_array: Array with initial labels (0 = unlabeled)
            district_mask: Binary mask defining valid area (1 = inside district)
            max_iterations: Maximum number of dilation iterations
            visualize: Whether to visualize the dilation process with OpenCV
            viz_interval: Show visualization every N iterations (default: 1)
            window_name: Name for the OpenCV visualization window

        Returns:
            Filled labeled array
        """
        result = labeled_array.copy()
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        
        # Setup visualization if enabled
        color_map = None
        if visualize:
            num_labels = len(np.unique(result[result > 0]))
            color_map = self._generate_color_map(num_labels)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info("Visualization enabled. Press 'q' to quit, 'p' to pause/resume, any other key to continue")

        paused = False
        
        for iteration in range(max_iterations):
            # Find unlabeled pixels within district
            unlabeled_mask = (result == 0) & (district_mask == 1)

            if not unlabeled_mask.any():
                logger.debug("All pixels labeled after %d iterations", iteration)
                if visualize:
                    self._visualize_labels(result, color_map, district_mask, iteration, window_name)
                    cv2.waitKey(1000)  # Show final result for 1 second
                break

            # Dilate each label
            changed = False
            for label in np.unique(result[result > 0]):
                # Create mask for this label
                label_mask = (result == label)

                # Dilate by one pixel
                dilated = ndimage.binary_dilation(label_mask, structure=structure)

                # Only assign to unlabeled pixels within district
                new_pixels = dilated & unlabeled_mask

                if new_pixels.any():
                    result[new_pixels] = label
                    changed = True
            
            # Visualize progress
            if visualize and (iteration % viz_interval == 0 or not changed):
                self._visualize_labels(result, color_map, district_mask, iteration, window_name)
                
                # Handle keyboard input
                while True:
                    key = cv2.waitKey(1 if not paused else 0) & 0xFF
                    
                    if key == ord('q'):  # Quit
                        logger.info("Visualization interrupted by user")
                        cv2.destroyWindow(window_name)
                        return result
                    elif key == ord('p'):  # Pause/Resume
                        paused = not paused
                        logger.info("Visualization %s", "paused" if paused else "resumed")
                    elif not paused:  # Continue if not paused
                        break

            if not changed:
                logger.debug("No changes after %d iterations", iteration)
                break

        # Check if there are still unlabeled pixels
        remaining = ((result == 0) & (district_mask == 1)).sum()
        if remaining > 0:
            logger.warning("Still have %d unlabeled pixels after dilation", remaining)
        
        # Cleanup visualization
        if visualize:
            cv2.destroyWindow(window_name)

        return result

    def extract_boundaries(
        self,
        voronoi: np.ndarray,
        district_mask: np.ndarray,
        original_buildings_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract centerline boundaries between different Voronoi regions.

        Uses morphological gradient to detect region changes, then applies
        thinning to get single-pixel centerlines.

        Args:
            voronoi: Labeled Voronoi partition array
            district_mask: Binary mask (1 inside district, 0 outside)
            original_buildings_mask: Binary mask of original buildings (to exclude from boundaries)

        Returns:
            Binary boundary array (1 = boundary centerline, 0 = not boundary)
        """
        logger.debug("Extracting Voronoi boundary centerlines")

        # Use morphological gradient to detect boundaries
        # This gives us a thick boundary between regions
        from scipy.ndimage import grey_dilation, grey_erosion

        # Apply morphological gradient (dilation - erosion)
        # This highlights edges between different labels
        dilated = grey_dilation(voronoi, size=(3, 3))
        eroded = grey_erosion(voronoi, size=(3, 3))

        # Boundary exists where dilation != erosion
        boundaries_thick = (dilated != eroded).astype(np.uint8)

        # Only keep boundaries inside district and outside buildings
        valid_mask = (district_mask == 1) & (original_buildings_mask == 0)
        boundaries_thick = boundaries_thick & valid_mask

        # Thin boundaries to single-pixel width using skeletonization
        from skimage.morphology import skeletonize

        # Apply skeletonization to get centerlines
        if boundaries_thick.any():
            boundaries = skeletonize(boundaries_thick).astype(np.uint8)
        else:
            boundaries = boundaries_thick

        num_boundary_pixels = boundaries.sum()
        logger.info("Extracted %d boundary centerline pixels", num_boundary_pixels)

        return boundaries

    def vectorize_boundaries(
        self,
        boundaries: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert boundary skeleton raster to vector line features.

        Uses a pixel-tracing approach to extract continuous line segments
        from the skeletonized boundary raster.

        Args:
            boundaries: Binary boundary skeleton array (1 = boundary)
            transform: Affine transformation from raster to world coordinates
            crs: Coordinate reference system
            district_attrs: Optional district attributes to add to features

        Returns:
            GeoDataFrame with boundary lines
        """
        logger.debug("Vectorizing boundary skeleton to line features")

        if not boundaries.any():
            logger.warning("No boundary pixels to vectorize")
            return gpd.GeoDataFrame(columns=['geometry', 'length'], crs=crs)

        lines = []
        visited = np.zeros_like(boundaries, dtype=bool)

        def pixel_to_coords(row, col):
            """Convert pixel coordinates to world coordinates."""
            x, y = transform * (col + 0.5, row + 0.5)
            return (x, y)

        def get_neighbors(row, col):
            """Get 8-connected neighbors."""
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < boundaries.shape[0] and 0 <= c < boundaries.shape[1]:
                        if boundaries[r, c] == 1 and not visited[r, c]:
                            neighbors.append((r, c))
            return neighbors

        def trace_line(start_row, start_col):
            """Trace a line from a starting pixel."""
            path = [(start_row, start_col)]
            visited[start_row, start_col] = True

            current = (start_row, start_col)
            while True:
                neighbors = get_neighbors(current[0], current[1])
                if not neighbors:
                    break
                # Choose the first unvisited neighbor
                next_pixel = neighbors[0]
                visited[next_pixel[0], next_pixel[1]] = True
                path.append(next_pixel)
                current = next_pixel

            return path

        # Find all boundary pixels
        boundary_pixels = np.argwhere(boundaries == 1)

        logger.debug("Found %d boundary pixels to trace", len(boundary_pixels))

        # Trace lines from unvisited boundary pixels
        for row, col in boundary_pixels:
            if not visited[row, col]:
                path = trace_line(row, col)

                if len(path) >= 2:  # Need at least 2 points for a line
                    # Convert pixel path to world coordinates
                    coords = [pixel_to_coords(r, c) for r, c in path]

                    # Create LineString
                    line = LineString(coords)

                    if line.is_valid and line.length > 0:
                        # Simplify line
                        line_simplified = line.simplify(
                            self.simplify_tolerance,
                            preserve_topology=True
                        )
                        if not line_simplified.is_empty and line_simplified.length > 0:
                            lines.append(line_simplified)

        if not lines:
            logger.warning("No boundary lines extracted after tracing")
            return gpd.GeoDataFrame(columns=['geometry', 'length'], crs=crs)

        logger.info("Traced %d boundary line segments", len(lines))

        # Try to merge connected lines
        merged = linemerge(lines)
        if merged.geom_type == 'LineString':
            lines = [merged]
        elif merged.geom_type == 'MultiLineString':
            lines = list(merged.geoms)
        logger.debug("Merged into %d line features", len(lines))

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs=crs)

        # Calculate length
        gdf['length'] = gdf.geometry.length

        # Add district attributes if provided
        if district_attrs:
            for key, value in district_attrs.items():
                if key not in gdf.columns:
                    gdf[key] = value

        logger.info("Created GeoDataFrame with %d boundary features, total length: %.2f m",
                   len(gdf), gdf['length'].sum())

        return gdf

    def generate_voronoi_boundaries(
        self,
        building_raster: np.ndarray,
        district_mask: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
        visualize: bool = False,
        viz_interval: int = 1,
    ) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Complete workflow: generate Voronoi diagram and extract boundaries.

        Args:
            building_raster: 2D array with building pixels > 0
            district_mask: Binary mask (1 inside district, 0 outside)
            transform: Affine transformation matrix
            crs: Coordinate reference system
            district_attrs: Optional district attributes
            visualize: Whether to visualize the dilation process with OpenCV
            viz_interval: Show visualization every N iterations (default: 1)

        Returns:
            Tuple of (boundary_lines_gdf, voronoi_partition_array)
        """
        # Generate Voronoi partition
        voronoi, original_buildings_mask = self.generate_voronoi_from_raster(
            building_raster, district_mask, visualize=visualize, viz_interval=viz_interval
        )

        # Extract boundaries (centerlines between regions, excluding buildings)
        boundaries = self.extract_boundaries(voronoi, district_mask, original_buildings_mask)

        # Vectorize boundaries
        boundary_gdf = self.vectorize_boundaries(
            boundaries, transform, crs, district_attrs
        )

        return boundary_gdf, voronoi

