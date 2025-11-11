"""Chunk splitter for large district processing."""

from typing import List, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from .utils.logger import get_logger

logger = get_logger(__name__)


def calculate_required_chunks(raster_shape: Tuple[int, int], max_size: int) -> bool:
    """
    Determine if chunking is needed based on raster dimensions.

    Args:
        raster_shape: Tuple of (height, width) in pixels
        max_size: Maximum allowed dimension size

    Returns:
        True if chunking is required, False otherwise
    """
    height, width = raster_shape
    return height > max_size or width > max_size


def estimate_raster_shape(district_geom, resolution: float = 1.0, buffer: float = 10.0) -> Tuple[int, int]:
    """
    Estimate raster dimensions from district geometry.

    Args:
        district_geom: Shapely geometry of the district
        resolution: Pixel size in meters
        buffer: Buffer around district bounds in meters

    Returns:
        Tuple of (height, width) in pixels
    """
    minx, miny, maxx, maxy = district_geom.bounds
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    return height, width


def split_district_adaptive(
    district_geom,
    buildings_gdf: gpd.GeoDataFrame,
    num_chunks: int = 4,
    overlap: int = 100,
    resolution: float = 1.0
) -> List[Tuple[Polygon, gpd.GeoDataFrame]]:
    """
    Split district into chunks based on building spatial distribution.

    Uses k-means clustering on building centroids to create spatially
    balanced chunks with roughly equal building counts.

    Args:
        district_geom: Shapely geometry of the district
        buildings_gdf: GeoDataFrame with building polygons
        num_chunks: Number of chunks to create (default: 4)
        overlap: Overlap between chunks in pixels
        resolution: Pixel size in meters (for converting overlap)

    Returns:
        List of tuples (chunk_geometry, buildings_in_chunk)
    """
    logger.info("Splitting district into %d adaptive chunks (overlap: %d pixels = %.1f m)",
                num_chunks, overlap, overlap * resolution)

    if len(buildings_gdf) == 0:
        logger.warning("No buildings found, returning single chunk")
        return [(district_geom, buildings_gdf)]

    # Extract building centroids
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in buildings_gdf.geometry])

    # Use k-means clustering to group buildings
    if len(buildings_gdf) < num_chunks:
        logger.warning("Fewer buildings (%d) than chunks (%d), using %d chunks",
                      len(buildings_gdf), num_chunks, len(buildings_gdf))
        num_chunks = len(buildings_gdf)

    kmeans = KMeans(n_clusters=num_chunks, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(centroids)

    # Create chunks based on clusters
    chunks = []
    overlap_meters = overlap * resolution

    for cluster_id in range(num_chunks):
        # Get buildings in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_buildings = buildings_gdf[cluster_mask].copy()

        if len(cluster_buildings) == 0:
            logger.warning("Cluster %d has no buildings, skipping", cluster_id)
            continue

        # Create bounding box around cluster buildings with overlap
        cluster_union = unary_union(cluster_buildings.geometry)
        minx, miny, maxx, maxy = cluster_union.bounds

        # Add overlap buffer
        minx -= overlap_meters
        miny -= overlap_meters
        maxx += overlap_meters
        maxy += overlap_meters

        # Clip to district bounds (extended slightly for safety)
        district_minx, district_miny, district_maxx, district_maxy = district_geom.bounds
        minx = max(minx, district_minx - overlap_meters)
        miny = max(miny, district_miny - overlap_meters)
        maxx = min(maxx, district_maxx + overlap_meters)
        maxy = min(maxy, district_maxy + overlap_meters)

        chunk_box = box(minx, miny, maxx, maxy)

        # Intersect with district geometry to get actual chunk
        chunk_geom = chunk_box.intersection(district_geom)

        if chunk_geom.is_empty:
            logger.warning("Chunk %d geometry is empty after intersection, skipping", cluster_id)
            continue

        # Get all buildings that intersect this chunk (including overlap regions)
        chunk_buildings_mask = buildings_gdf.geometry.intersects(chunk_geom)
        chunk_buildings = buildings_gdf[chunk_buildings_mask].copy()

        logger.info("Chunk %d: %d buildings (%.1f%%), bounds: (%.1f, %.1f) to (%.1f, %.1f)",
                   cluster_id, len(chunk_buildings), 
                   len(chunk_buildings) / len(buildings_gdf) * 100,
                   minx, miny, maxx, maxy)

        chunks.append((chunk_geom, chunk_buildings))

    logger.info("Created %d chunks covering %d total buildings",
                len(chunks), len(buildings_gdf))

    return chunks

