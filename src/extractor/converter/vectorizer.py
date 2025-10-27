"""Vectorizer for converting segmentation results to vector format."""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio import features as rio_features
from affine import Affine
from ..utils.logger import get_logger

logger = get_logger()


class Vectorizer:
    """Class for vectorizing raster segmentation results."""

    def __init__(self, simplify_tolerance: float = 1.0):
        """
        Initialize vectorizer.

        Args:
            simplify_tolerance: Tolerance for Douglas-Peucker simplification (meters)
        """
        self.simplify_tolerance = simplify_tolerance

    def vectorize_segments(
        self,
        segmentation: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert segmentation raster to vector polygons.

        Args:
            segmentation: Labeled raster array (H, W)
            transform: Affine transformation from raster to world coordinates
            crs: Coordinate reference system
            district_attrs: Optional district attributes to copy to each segment

        Returns:
            GeoDataFrame with segmented polygons
        """
        logger.debug("Vectorizing segmentation results")

        polygons = []
        cluster_ids = []

        # Extract shapes for each cluster
        mask = segmentation > 0
        unique_labels = np.unique(segmentation[mask])

        for label in unique_labels:
            if label == 0:
                continue

            # Create binary mask for this cluster
            cluster_mask = (segmentation == label).astype(np.uint8)

            # Extract shapes
            shapes = list(
                rio_features.shapes(
                    cluster_mask, mask=cluster_mask > 0, transform=transform
                )
            )

            # Combine all polygons for this cluster
            cluster_polygons = []
            for geom, value in shapes:
                if value == 1:
                    poly = shape(geom)
                    if poly.is_valid and not poly.is_empty:
                        # Simplify polygon
                        poly_simplified = poly.simplify(
                            self.simplify_tolerance, preserve_topology=True
                        )
                        if not poly_simplified.is_empty:
                            cluster_polygons.append(poly_simplified)

            if cluster_polygons:
                # Merge overlapping polygons
                merged_poly = unary_union(cluster_polygons)
                polygons.append(merged_poly)
                cluster_ids.append(int(label))

        logger.debug("Vectorized %d segments", len(polygons))

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"cluster_id": cluster_ids, "geometry": polygons}, crs=crs
        )

        # Calculate area
        gdf["area"] = gdf.geometry.area

        # Add district attributes if provided
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
        """
        Count buildings within each segment.

        Args:
            segments_gdf: GeoDataFrame with segmented regions
            buildings_gdf: GeoDataFrame with building polygons

        Returns:
            Updated GeoDataFrame with building_count column
        """
        logger.debug("Counting buildings in each segment")

        building_counts = []
        for _, segment in segments_gdf.iterrows():
            # Count buildings intersecting this segment
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
        """
        Merge multiple segment GeoDataFrames into one.

        Args:
            segment_gdfs: List of GeoDataFrames to merge
            continuous_ids: Whether to reassign continuous cluster IDs

        Returns:
            Merged GeoDataFrame
        """
        if not segment_gdfs:
            logger.warning("No segments to merge")
            return gpd.GeoDataFrame()

        logger.info("Merging %d segment collections", len(segment_gdfs))

        # Concatenate all GeoDataFrames
        merged = gpd.GeoDataFrame(pd.concat(segment_gdfs, ignore_index=True))

        if continuous_ids:
            # Reassign continuous cluster IDs
            merged["cluster_id"] = range(1, len(merged) + 1)
            logger.debug("Reassigned continuous IDs: 1 to %d", len(merged))

        return merged

