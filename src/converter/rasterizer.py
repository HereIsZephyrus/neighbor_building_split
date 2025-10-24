"""Rasterizer for converting vector geometries to raster format."""

from typing import Tuple
import numpy as np
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
from affine import Affine
from ..utils.logger import get_logger

logger = get_logger()


class Rasterizer:
    """Class for rasterizing vector data to raster format."""

    def __init__(self, resolution: float = 1.0):
        """
        Initialize rasterizer.

        Args:
            resolution: Pixel size in meters (default: 1.0)
        """
        self.resolution = resolution

    def rasterize_buildings(
        self,
        district_geometry,
        buildings: gpd.GeoDataFrame,
        buffer: float = 10.0,
    ) -> Tuple[np.ndarray, Affine, Tuple[float, float, float, float]]:
        """
        Rasterize buildings within a district with floor values.

        Args:
            district_geometry: Shapely geometry of the district
            buildings: GeoDataFrame with building polygons and 'floor' attribute
            buffer: Buffer around district bounds in meters (default: 10.0)

        Returns:
            Tuple of (raster_array, affine_transform, bounds)
            - raster_array: 2D numpy array with background=0, building pixels=floor value
            - affine_transform: Affine transformation matrix
            - bounds: (minx, miny, maxx, maxy)
        """
        # Get bounds and add buffer
        minx, miny, maxx, maxy = district_geometry.bounds
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
        bounds = (minx, miny, maxx, maxy)

        # Calculate raster dimensions
        width = int(np.ceil((maxx - minx) / self.resolution))
        height = int(np.ceil((maxy - miny) / self.resolution))

        logger.debug("Rasterizing to %dx%d pixels at %sm resolution",
                     width, height, self.resolution)

        # Create affine transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Initialize raster with zeros
        raster = np.zeros((height, width), dtype=np.float32)

        if len(buildings) == 0:
            logger.warning("No buildings to rasterize")
            return raster, transform, bounds

        # Prepare geometries and values for rasterization
        shapes = []
        for _, building in buildings.iterrows():
            floor_value = building.get("floor", 1.0)
            if floor_value > 0:  # Only rasterize buildings with positive floor values
                shapes.append((building.geometry, float(floor_value)))

        if shapes:
            # Rasterize buildings
            rasterized = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.float32,
            )
            raster = rasterized

        logger.debug(
            "Rasterized %d buildings, coverage: %.2f%%",
            len(shapes), (raster > 0).sum() / raster.size * 100
        )

        return raster, transform, bounds

    def rasterize_district_mask(
        self,
        district_geometry,
        transform: Affine,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create binary mask for district geometry.

        Args:
            district_geometry: Shapely geometry of the district
            transform: Affine transformation matrix
            shape: Raster shape (height, width)

        Returns:
            Binary mask array (1 inside district, 0 outside)
        """
        mask = features.rasterize(
            shapes=[(district_geometry, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )

        logger.debug(
            "Created district mask, coverage: %.2f%%",
            mask.sum() / mask.size * 100
        )

        return mask

