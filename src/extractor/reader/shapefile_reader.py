"""Shapefile reader for district and building data."""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ShapefileReader:
    """Load and process district and building shapefiles."""

    TARGET_CRS = "EPSG:32650"

    def __init__(self, district_path: Path, building_path: Path):
        """Initialize with paths to district and building shapefiles."""
        self.district_path = district_path
        self.building_path = building_path
        self._districts = None
        self._buildings = None

    def load_districts(self) -> gpd.GeoDataFrame:
        """Load district shapefile and reproject to EPSG:32650."""
        if not self.district_path.exists():
            raise FileNotFoundError(f"District file not found: {self.district_path}")

        logger.debug("Loading districts from %s", self.district_path)
        gdf = gpd.read_file(self.district_path)

        if gdf.crs is None:
            logger.warning("District CRS missing, assuming EPSG:32650")
            gdf.set_crs(self.TARGET_CRS, inplace=True)
        elif gdf.crs != self.TARGET_CRS:
            logger.debug("Reprojecting districts to %s", self.TARGET_CRS)
            gdf = gdf.to_crs(self.TARGET_CRS)

        self._districts = gdf
        logger.info("Loaded %d districts", len(gdf))
        return gdf

    def load_buildings(self) -> gpd.GeoDataFrame:
        """Load building shapefile and reproject to EPSG:32650."""
        if not self.building_path.exists():
            raise FileNotFoundError(f"Building file not found: {self.building_path}")

        logger.debug("Loading buildings from %s", self.building_path)
        gdf = gpd.read_file(self.building_path)

        if gdf.crs is None:
            logger.warning("Building CRS missing, assuming EPSG:32650")
            gdf.set_crs(self.TARGET_CRS, inplace=True)
        elif gdf.crs != self.TARGET_CRS:
            logger.debug("Reprojecting buildings to %s", self.TARGET_CRS)
            gdf = gdf.to_crs(self.TARGET_CRS)

        self._buildings = gdf
        logger.info("Loaded %d buildings", len(gdf))
        return gdf

    def get_buildings_in_district(
        self, district_geometry
    ) -> gpd.GeoDataFrame:
        """Clip buildings to district boundary and fix invalid geometries."""
        if self._buildings is None:
            self.load_buildings()

        bbox = box(*district_geometry.bounds)
        buildings_bbox = self._buildings[self._buildings.intersects(bbox)]

        if len(buildings_bbox) == 0:
            logger.warning("No buildings in district bbox")
            return gpd.GeoDataFrame(columns=self._buildings.columns, crs=self._buildings.crs)

        district_gdf = gpd.GeoDataFrame([1], geometry=[district_geometry], crs=self._buildings.crs)
        buildings_clipped = gpd.clip(buildings_bbox, district_gdf)

        empty_count = buildings_clipped.geometry.is_empty.sum()
        if empty_count > 0:
            logger.debug("Filtering %d empty geometries", empty_count)
            buildings_clipped = buildings_clipped[~buildings_clipped.geometry.is_empty]

        invalid_count = (~buildings_clipped.geometry.is_valid).sum()
        if invalid_count > 0:
            logger.debug("Fixing %d invalid geometries", invalid_count)
            buildings_clipped.geometry = buildings_clipped.geometry.buffer(0)

        logger.debug("%d buildings clipped from %d in bbox", 
                    len(buildings_clipped), len(buildings_bbox))

        return buildings_clipped

    @property
    def districts(self) -> gpd.GeoDataFrame:
        """Get loaded districts (load if not already loaded)."""
        if self._districts is None:
            self.load_districts()
        return self._districts

    @property
    def buildings(self) -> gpd.GeoDataFrame:
        """Get loaded buildings (load if not already loaded)."""
        if self._buildings is None:
            self.load_buildings()
        return self._buildings

