"""Voronoi diagram generator using dilation method."""

from typing import Tuple, Optional
import numpy as np
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import shape, LineString
from shapely.ops import linemerge
from rasterio import features as rio_features
from affine import Affine
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
    ) -> np.ndarray:
        """
        Generate Voronoi partition from building raster using dilation method.

        This method:
        1. Identifies connected components (buildings) in the raster
        2. Uses distance transform and watershed-like dilation to partition space
        3. Respects district boundaries

        Args:
            building_raster: 2D array with building pixels > 0, background = 0, outside = -999
            district_mask: Binary mask (1 inside district, 0 outside)

        Returns:
            Voronoi partition array with each region labeled uniquely (0 for outside)
        """
        logger.debug("Generating Voronoi diagram using dilation method")

        # Create binary mask for buildings (any positive value is a building)
        building_binary = (building_raster > 0).astype(np.uint8)

        # Identify connected components (buildings as seeds)
        # Use 8-connectivity to consider diagonal neighbors
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        labeled_buildings, num_features = ndimage.label(building_binary, structure=structure)

        logger.info("Identified %d building connected components", num_features)

        if num_features == 0:
            logger.warning("No buildings found, returning empty Voronoi diagram")
            return np.zeros_like(building_raster, dtype=np.int32)

        # Initialize Voronoi partition with building labels
        voronoi = labeled_buildings.copy()

        # Create mask for areas that need to be filled (inside district but not building)
        fill_mask = (district_mask == 1) & (building_binary == 0)

        # For each unlabeled pixel in the district, find nearest building label
        # This is equivalent to Voronoi tessellation using dilation
        if fill_mask.any():
            # Use dilation-based approach for better control
            # Iteratively expand labeled regions
            voronoi = self._dilate_labels(voronoi, district_mask, max_iterations=10000)

        logger.debug("Voronoi diagram generated with %d regions", num_features)

        return voronoi

    def _dilate_labels(
        self,
        labeled_array: np.ndarray,
        district_mask: np.ndarray,
        max_iterations: int = 10000,
    ) -> np.ndarray:
        """
        Dilate labeled regions to fill entire district using morphological dilation.

        Args:
            labeled_array: Array with initial labels (0 = unlabeled)
            district_mask: Binary mask defining valid area (1 = inside district)
            max_iterations: Maximum number of dilation iterations

        Returns:
            Filled labeled array
        """
        result = labeled_array.copy()
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity

        for iteration in range(max_iterations):
            # Find unlabeled pixels within district
            unlabeled_mask = (result == 0) & (district_mask == 1)
            
            if not unlabeled_mask.any():
                logger.debug("All pixels labeled after %d iterations", iteration)
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

            if not changed:
                logger.debug("No changes after %d iterations", iteration)
                break

        # Check if there are still unlabeled pixels
        remaining = ((result == 0) & (district_mask == 1)).sum()
        if remaining > 0:
            logger.warning("Still have %d unlabeled pixels after dilation", remaining)

        return result

    def extract_boundaries(
        self,
        voronoi: np.ndarray,
        district_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract boundaries between different Voronoi regions.

        Args:
            voronoi: Labeled Voronoi partition array
            district_mask: Binary mask (1 inside district, 0 outside)

        Returns:
            Binary boundary array (1 = boundary, 0 = not boundary)
        """
        logger.debug("Extracting Voronoi boundaries")

        height, width = voronoi.shape
        boundaries = np.zeros((height, width), dtype=np.uint8)

        # Create valid mask (inside district and labeled)
        valid_mask = (district_mask == 1) & (voronoi > 0)

        # Check horizontal neighbors
        for i in range(height):
            for j in range(width - 1):
                if valid_mask[i, j] and valid_mask[i, j + 1]:
                    if voronoi[i, j] != voronoi[i, j + 1]:
                        boundaries[i, j] = 1
                        boundaries[i, j + 1] = 1

        # Check vertical neighbors
        for i in range(height - 1):
            for j in range(width):
                if valid_mask[i, j] and valid_mask[i + 1, j]:
                    if voronoi[i, j] != voronoi[i + 1, j]:
                        boundaries[i, j] = 1
                        boundaries[i + 1, j] = 1

        # Optional: thin the boundaries to single-pixel width
        # boundaries = ndimage.binary_erosion(boundaries, iterations=1).astype(np.uint8)

        num_boundary_pixels = boundaries.sum()
        logger.info("Extracted %d boundary pixels", num_boundary_pixels)

        return boundaries

    def vectorize_boundaries(
        self,
        boundaries: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert boundary raster to vector line features.

        Args:
            boundaries: Binary boundary array (1 = boundary)
            transform: Affine transformation from raster to world coordinates
            crs: Coordinate reference system
            district_attrs: Optional district attributes to add to features

        Returns:
            GeoDataFrame with boundary lines
        """
        logger.debug("Vectorizing boundaries to line features")

        lines = []

        # Extract shapes from boundary raster
        # We treat boundaries as polygons first, then extract their outlines
        shapes = list(
            rio_features.shapes(
                boundaries.astype(np.uint8),
                mask=boundaries > 0,
                transform=transform,
                connectivity=8
            )
        )

        logger.debug("Found %d boundary shapes", len(shapes))

        for geom_dict, value in shapes:
            if value == 1:
                # Convert to shapely geometry
                poly = shape(geom_dict)
                
                if poly.is_valid and not poly.is_empty:
                    # Extract the boundary (exterior) as a line
                    if poly.geom_type == 'Polygon':
                        line = LineString(poly.exterior.coords)
                        if line.length > 0:
                            # Simplify line
                            line_simplified = line.simplify(
                                self.simplify_tolerance,
                                preserve_topology=True
                            )
                            if not line_simplified.is_empty and line_simplified.length > 0:
                                lines.append(line_simplified)
                    elif poly.geom_type == 'MultiPolygon':
                        for sub_poly in poly.geoms:
                            line = LineString(sub_poly.exterior.coords)
                            if line.length > 0:
                                line_simplified = line.simplify(
                                    self.simplify_tolerance,
                                    preserve_topology=True
                                )
                                if not line_simplified.is_empty and line_simplified.length > 0:
                                    lines.append(line_simplified)

        if not lines:
            logger.warning("No boundary lines extracted")
            return gpd.GeoDataFrame(columns=['geometry', 'length'], crs=crs)

        logger.info("Extracted %d boundary line segments", len(lines))

        # Try to merge connected lines
        try:
            merged = linemerge(lines)
            if merged.geom_type == 'LineString':
                lines = [merged]
            elif merged.geom_type == 'MultiLineString':
                lines = list(merged.geoms)
            logger.debug("Merged into %d line features", len(lines))
        except Exception as exc:
            logger.warning("Could not merge lines: %s", exc)

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
    ) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Complete workflow: generate Voronoi diagram and extract boundaries.

        Args:
            building_raster: 2D array with building pixels > 0
            district_mask: Binary mask (1 inside district, 0 outside)
            transform: Affine transformation matrix
            crs: Coordinate reference system
            district_attrs: Optional district attributes

        Returns:
            Tuple of (boundary_lines_gdf, voronoi_partition_array)
        """
        # Generate Voronoi partition
        voronoi = self.generate_voronoi_from_raster(building_raster, district_mask)

        # Extract boundaries
        boundaries = self.extract_boundaries(voronoi, district_mask)

        # Vectorize boundaries
        boundary_gdf = self.vectorize_boundaries(
            boundaries, transform, crs, district_attrs
        )

        return boundary_gdf, voronoi

