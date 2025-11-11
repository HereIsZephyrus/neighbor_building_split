"""Convert vector geometries to raster format."""

from typing import Tuple
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from affine import Affine
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Rasterizer:
    """Rasterize vector geometries to pixel arrays."""

    def __init__(self, resolution: float = 1.0):
        """Initialize with pixel resolution in meters."""
        self.resolution = resolution

    def rasterize_buildings(
        self,
        district_geometry,
        buildings: gpd.GeoDataFrame,
        buffer: float = 10.0,
    ) -> Tuple[np.ndarray, Affine, Tuple[float, float, float, float]]:
        """
        Rasterize buildings with IDs as pixel values.
        
        Returns (raster, transform, bounds) where raster pixels are:
        building_id (>0), background (0), or NoData (-999).
        """
        minx, miny, maxx, maxy = district_geometry.bounds
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
        bounds = (minx, miny, maxx, maxy)

        width = int(np.ceil((maxx - minx) / self.resolution))
        height = int(np.ceil((maxy - miny) / self.resolution))

        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        raster = np.full((height, width), -999, dtype=np.int32)
        district_mask = self.rasterize_district_mask(district_geometry, transform, (height, width))
        raster[district_mask == 1] = 0

        if len(buildings) == 0:
            logger.warning("No buildings to rasterize")
            return raster, transform, bounds

        shapes = []
        skipped_invalid = 0
        missing_id = 0

        id_field = None
        for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'fid']:
            if possible_id in buildings.columns:
                id_field = possible_id
                break

        for idx, building in buildings.iterrows():
            if building.geometry is None or building.geometry.is_empty:
                skipped_invalid += 1
                continue

            if id_field is not None:
                building_id = building.get(id_field)
                if building_id is None or (isinstance(building_id, float) and math.isnan(building_id)):
                    building_id = int(idx) + 1 if isinstance(idx, int) else hash(str(idx)) % 2147483647
                    missing_id += 1
                else:
                    building_id = int(building_id)
            else:
                building_id = int(idx) + 1 if isinstance(idx, int) else hash(str(idx)) % 2147483647
                if missing_id == 0:
                    logger.warning("No ID field, using indices as IDs")
                missing_id += 1

            shapes.append((building.geometry, building_id))

        if skipped_invalid > 0:
            logger.warning("Skipped %d invalid geometries", skipped_invalid)
        if missing_id > 0 and id_field is not None:
            logger.debug("Generated IDs for %d buildings", missing_id)

        if shapes:
            rasterized = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.int32,
            )
            raster[(district_mask == 1) & (rasterized > 0)] = rasterized[(district_mask == 1) & (rasterized > 0)]

        building_pixels = (raster > 0).sum()
        district_pixels = (district_mask == 1).sum()

        logger.debug("%dx%d raster: %d buildings → %d pixels (%.1f%%)",
            width, height, len(shapes), building_pixels,
            (building_pixels / district_pixels * 100) if district_pixels > 0 else 0
        )

        return raster, transform, bounds

    def rasterize_district_mask(
        self,
        district_geometry,
        transform: Affine,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create binary mask for district boundary (1=inside, 0=outside)."""
        mask = features.rasterize(
            shapes=[(district_geometry, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

        return mask

    def save_raster_as_tif(
        self,
        raster: np.ndarray,
        transform: Affine,
        output_path: Path,
        crs: str = "EPSG:32650",
        nodata: int = -999,
    ) -> None:
        """Save raster array as GeoTIFF."""
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

        logger.debug("Saved raster to %s", output_path)

