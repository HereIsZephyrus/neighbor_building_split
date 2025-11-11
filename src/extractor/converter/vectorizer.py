"""Convert raster segmentation results to vector format."""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio import features as rio_features
from affine import Affine
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Vectorizer:
    """Vectorize raster segmentation to polygons."""

    def __init__(self, simplify_tolerance: float = 1.0):
        """Initialize with simplification tolerance in meters."""
        self.simplify_tolerance = simplify_tolerance

    def vectorize_segments(
        self,
        segmentation: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """Convert segmentation raster to vector polygons."""
        polygons = []
        cluster_ids = []

        mask = segmentation > 0
        unique_labels = np.unique(segmentation[mask])

        for label in unique_labels:
            if label == 0:
                continue

            cluster_mask = (segmentation == label).astype(np.uint8)

            shapes = list(
                rio_features.shapes(
                    cluster_mask, mask=cluster_mask > 0, transform=transform
                )
            )

            cluster_polygons = []
            for geom, value in shapes:
                if value == 1:
                    poly = shape(geom)
                    if poly.is_valid and not poly.is_empty:
                        poly_simplified = poly.simplify(
                            self.simplify_tolerance, preserve_topology=True
                        )
                        if not poly_simplified.is_empty:
                            cluster_polygons.append(poly_simplified)

            if cluster_polygons:
                merged_poly = unary_union(cluster_polygons)
                polygons.append(merged_poly)
                cluster_ids.append(int(label))

        logger.debug("Vectorized %d segments", len(polygons))

        gdf = gpd.GeoDataFrame(
            {"cluster_id": cluster_ids, "geometry": polygons}, crs=crs
        )

        gdf["area"] = gdf.geometry.area

        if district_attrs:
            for key, value in district_attrs.items():
                if key not in gdf.columns:
                    gdf[key] = value

        return gdf

    def count_buildings_in_segments(
        self,
        segments_gdf: gpd.GeoDataFrame,
        buildings_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Count buildings within each segment polygon."""
        building_counts = []
        for _, segment in segments_gdf.iterrows():
            intersecting = buildings_gdf[
                buildings_gdf.intersects(segment.geometry)
            ]
            building_counts.append(len(intersecting))

        segments_gdf = segments_gdf.copy()
        segments_gdf["building_count"] = building_counts

        return segments_gdf

    def merge_segments(
        self, segment_gdfs: list, continuous_ids: bool = True
    ) -> gpd.GeoDataFrame:
        """Merge multiple segment GeoDataFrames into one."""
        if not segment_gdfs:
            logger.warning("No segments to merge")
            return gpd.GeoDataFrame()

        merged = gpd.GeoDataFrame(pd.concat(segment_gdfs, ignore_index=True))

        if continuous_ids:
            merged["cluster_id"] = range(1, len(merged) + 1)
            logger.debug("Reassigned IDs: 1 to %d", len(merged))

        return merged

