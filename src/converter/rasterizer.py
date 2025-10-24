"""Rasterizer for converting vector geometries to raster format."""

from typing import Tuple
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from affine import Affine
from pathlib import Path
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
        Rasterize buildings within a district with Floor values.
        Areas outside the district are set to -999 (NoData).

        Args:
            district_geometry: Shapely geometry of the district
            buildings: GeoDataFrame with building polygons and 'Floor' attribute
            buffer: Buffer around district bounds in meters (default: 10.0)

        Returns:
            Tuple of (raster_array, affine_transform, bounds)
            - raster_array: 2D numpy array with background=0, building pixels=Floor value, outside district=-999
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

        # Initialize raster with NoData values (-999) for areas outside district
        raster = np.full((height, width), -999.0, dtype=np.float32)
        
        # Create district mask to identify areas inside the district
        district_mask = self.rasterize_district_mask(district_geometry, transform, (height, width))
        
        # Set areas inside district to 0 (background)
        raster[district_mask == 1] = 0.0

        if len(buildings) == 0:
            logger.warning("No buildings to rasterize")
            return raster, transform, bounds

        # Prepare geometries and values for rasterization
        shapes = []
        for _, building in buildings.iterrows():
            floor_value = building.get("Floor", 1.0)
            if floor_value > 0:  # Only rasterize buildings with positive Floor values
                shapes.append((building.geometry, float(floor_value)))

        if shapes:
            # Rasterize buildings (only in areas where district_mask == 1)
            rasterized = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.float32,
            )
            # Only update pixels that are inside the district
            raster[(district_mask == 1) & (rasterized > 0)] = rasterized[(district_mask == 1) & (rasterized > 0)]

        logger.debug(
            "Rasterized %d buildings, coverage: %.2f%% (inside district: %.2f%%)",
            len(shapes), 
            (raster > 0).sum() / raster.size * 100,
            (district_mask == 1).sum() / raster.size * 100
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

    def save_raster_as_tif(
        self,
        raster: np.ndarray,
        transform: Affine,
        output_path: Path,
        crs: str = "EPSG:32650",
        nodata: float = -999.0,
    ) -> None:
        """
        Save raster array as GeoTIFF file.

        Args:
            raster: 2D numpy array to save
            transform: Affine transformation matrix
            output_path: Path where to save the TIF file
            crs: Coordinate reference system (default: EPSG:32650)
            nodata: NoData value (default: -999.0)
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(raster, 1)

        logger.info("Saved raster to %s", output_path)

