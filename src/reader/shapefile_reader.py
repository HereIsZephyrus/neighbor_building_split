"""Shapefile reader for district and building data."""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from ..utils.logger import get_logger

logger = get_logger()


class ShapefileReader:
    """Reader class for loading and processing shapefiles."""

    TARGET_CRS = "EPSG:32650"

    def __init__(self, district_path: Path, building_path: Path):
        """
        Initialize shapefile reader.

        Args:
            district_path: Path to district shapefile
            building_path: Path to building shapefile
        """
        self.district_path = district_path
        self.building_path = building_path
        self._districts = None
        self._buildings = None

    def load_districts(self) -> gpd.GeoDataFrame:
        """
        Load district shapefile and reproject to target CRS.

        Returns:
            GeoDataFrame with district polygons

        Raises:
            FileNotFoundError: If district file doesn't exist
        """
        if not self.district_path.exists():
            raise FileNotFoundError(f"District file not found: {self.district_path}")

        logger.info("Loading districts from %s", self.district_path)
        gdf = gpd.read_file(self.district_path)

        if gdf.crs is None:
            logger.warning("District CRS is None, assuming EPSG:32650")
            gdf.set_crs(self.TARGET_CRS, inplace=True)
        elif gdf.crs != self.TARGET_CRS:
            logger.info("Reprojecting districts from %s to %s", gdf.crs, self.TARGET_CRS)
            gdf = gdf.to_crs(self.TARGET_CRS)

        self._districts = gdf
        logger.info("Loaded %d district features", len(gdf))
        return gdf

    def load_buildings(self) -> gpd.GeoDataFrame:
        """
        Load building shapefile and reproject to target CRS.

        Returns:
            GeoDataFrame with building polygons

        Raises:
            FileNotFoundError: If building file doesn't exist
            ValueError: If 'Floor' attribute is missing
        """
        if not self.building_path.exists():
            raise FileNotFoundError(f"Building file not found: {self.building_path}")

        logger.info("Loading buildings from %s", self.building_path)
        gdf = gpd.read_file(self.building_path)

        if "Floor" not in gdf.columns:
            raise ValueError("Building shapefile must contain 'Floor' attribute")

        if gdf.crs is None:
            logger.warning("Building CRS is None, assuming EPSG:32650")
            gdf.set_crs(self.TARGET_CRS, inplace=True)
        elif gdf.crs != self.TARGET_CRS:
            logger.info("Reprojecting buildings from %s to %s", gdf.crs, self.TARGET_CRS)
            gdf = gdf.to_crs(self.TARGET_CRS)

        self._buildings = gdf
        logger.info("Loaded %d building features", len(gdf))
        return gdf

    def get_buildings_in_district(
        self, district_geometry
    ) -> gpd.GeoDataFrame:
        """
        Clip buildings to district geometry and return only buildings within the district.

        Args:
            district_geometry: Shapely geometry of the district

        Returns:
            GeoDataFrame with buildings clipped to the district boundary
        """
        if self._buildings is None:
            self.load_buildings()

        # Spatial filter using bounding box first for efficiency
        bbox = box(*district_geometry.bounds)
        buildings_bbox = self._buildings[self._buildings.intersects(bbox)]

        if len(buildings_bbox) == 0:
            logger.debug("No buildings found in district bounding box")
            return gpd.GeoDataFrame(columns=self._buildings.columns, crs=self._buildings.crs)

        # Clip buildings to district geometry
        # Create a GeoDataFrame with the district geometry for clipping
        district_gdf = gpd.GeoDataFrame([1], geometry=[district_geometry], crs=self._buildings.crs)

        # Perform spatial clip operation
        buildings_clipped = gpd.clip(buildings_bbox, district_gdf)

        # Filter out empty geometries that might result from clipping
        buildings_clipped = buildings_clipped[~buildings_clipped.geometry.is_empty]

        logger.debug(
            "Found %d buildings in district (from %d in bbox, %d total)",
            len(buildings_clipped), len(buildings_bbox), len(self._buildings)
        )

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

