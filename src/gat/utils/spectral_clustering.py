"""Spectral clustering utilities for building clustering.

Combines GAT embeddings, building features, and adjacency matrix
to perform spectral clustering and assign GAT labels to clusters.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

from .logger import get_logger

logger = get_logger()


def compute_affinity_matrix(
    embeddings: np.ndarray,
    features: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    embedding_weight: float = 0.5,
    feature_weight: float = 0.3,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0
) -> np.ndarray:
    """
    Compute affinity matrix by combining GAT embeddings, building features, and adjacency.

    Args:
        embeddings: GAT node embeddings (N, D_emb)
        features: Original building features (N, D_feat)
        adjacency_matrix: Distance-based adjacency matrix (N, N)
        embedding_weight: Weight for embedding similarity
        feature_weight: Weight for feature similarity
        distance_weight: Weight for distance-based adjacency
        distance_scale: Scale factor for distance conversion

    Returns:
        affinity_matrix: Combined affinity matrix (N, N)
    """
    logger.debug(
        "Computing affinity matrix: embeddings=%s, features=%s, adjacency=%s",
        embeddings.shape, features.shape, adjacency_matrix.shape
    )

    # 1. Embedding-based similarity (cosine similarity)
    embedding_sim = cosine_similarity(embeddings)
    # Normalize to [0, 1]
    embedding_sim = (embedding_sim + 1) / 2
    logger.debug("Embedding similarity: min=%.4f, max=%.4f", embedding_sim.min(), embedding_sim.max())

    # 2. Feature-based similarity (cosine similarity)
    feature_sim = cosine_similarity(features)
    # Normalize to [0, 1]
    feature_sim = (feature_sim + 1) / 2
    logger.debug("Feature similarity: min=%.4f, max=%.4f", feature_sim.min(), feature_sim.max())

    # 3. Distance-based affinity from adjacency matrix
    # Adjacency matrix contains distances (smaller = closer)
    # Convert to similarity: similarity = exp(-distance / scale)
    distance_matrix = adjacency_matrix.values

    # Handle cases where distance is 0 or very small
    distance_affinity = np.exp(-distance_matrix / distance_scale)

    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(distance_affinity, 1.0)

    # Zero out non-adjacent nodes (where original adjacency was 0)
    distance_affinity[distance_matrix == 0] = 0

    logger.debug("Distance affinity: min=%.4f, max=%.4f", 
                 distance_affinity[distance_affinity > 0].min(), distance_affinity.max())

    # 4. Combine all three similarity measures
    affinity = (
        embedding_weight * embedding_sim +
        feature_weight * feature_sim +
        distance_weight * distance_affinity
    )

    # Normalize combined affinity to [0, 1]
    affinity_min = affinity.min()
    affinity_max = affinity.max()
    if affinity_max > affinity_min:
        affinity = (affinity - affinity_min) / (affinity_max - affinity_min)

    # Ensure symmetry
    affinity = (affinity + affinity.T) / 2

    # Set diagonal to 1
    np.fill_diagonal(affinity, 1.0)

    logger.debug("Final affinity matrix: min=%.4f, max=%.4f, mean=%.4f", 
                 affinity.min(), affinity.max(), affinity.mean())

    return affinity


def spectral_cluster(
    affinity_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Perform spectral clustering on affinity matrix.

    Args:
        affinity_matrix: Affinity/similarity matrix (N, N)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        cluster_labels: Cluster assignments (N,)
    """
    logger.info("Performing spectral clustering: n_clusters=%d, matrix_shape=%s", n_clusters, affinity_matrix.shape)

    # Spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state,
        n_init=10,
        assign_labels='kmeans'
    )

    cluster_labels = clustering.fit_predict(affinity_matrix)

    logger.info("Spectral clustering completed: %d clusters assigned", len(np.unique(cluster_labels)))

    return cluster_labels


def assign_labels_to_clusters(
    cluster_assignments: np.ndarray,
    gat_labels: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    Assign GAT predicted labels to spectral clusters based on majority voting.

    Args:
        cluster_assignments: Spectral cluster assignments (N,)
        gat_labels: GAT predicted labels (N,)

    Returns:
        cluster_to_label: Mapping from cluster ID to GAT label (dict)
        final_labels: Final label for each node based on cluster assignment (N,)
    """
    logger.debug("Assigning GAT labels to spectral clusters via majority voting")

    unique_clusters = np.unique(cluster_assignments)
    cluster_to_label = {}

    for cluster_id in unique_clusters:
        # Find all nodes in this cluster
        mask = cluster_assignments == cluster_id
        cluster_gat_labels = gat_labels[mask]

        # Majority vote: most common GAT label in this cluster
        unique_labels, counts = np.unique(cluster_gat_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]

        cluster_to_label[int(cluster_id)] = int(majority_label)

        logger.debug(
            "Cluster %d: %d nodes, GAT label distribution=%s, assigned label=%d",
            cluster_id, mask.sum(), dict(zip(unique_labels.tolist(), counts.tolist())), majority_label
        )

    # Create final labels array
    final_labels = np.array([cluster_to_label[int(c)] for c in cluster_assignments])

    logger.info("Assigned GAT labels to %d clusters", len(cluster_to_label))

    return cluster_to_label, final_labels


def estimate_optimal_clusters(
    affinity_matrix: np.ndarray,
    max_clusters: int = 15,
    min_clusters: int = 2,
    oversample_factor: float = 1.5
) -> int:
    """
    Estimate optimal number of clusters using eigenvalue analysis with oversampling.

    Args:
        affinity_matrix: Affinity matrix (N, N)
        max_clusters: Maximum number of clusters to consider
        min_clusters: Minimum number of clusters
        oversample_factor: Multiplier for oversampling clusters (default 1.5)

    Returns:
        optimal_k: Estimated optimal number of clusters (with oversampling)
    """
    from scipy.linalg import eigh

    logger.debug("Estimating optimal number of clusters via eigenvalue analysis")

    # Compute Laplacian
    degree_matrix = np.diag(affinity_matrix.sum(axis=1))
    laplacian = degree_matrix - affinity_matrix

    # Compute eigenvalues
    eigenvalues, _ = eigh(laplacian)

    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)

    # Find eigengap (largest gap in first max_clusters eigenvalues)
    max_clusters = min(max_clusters, len(eigenvalues) - 1)
    eigengaps = np.diff(eigenvalues[:max_clusters + 1])

    optimal_k_base = np.argmax(eigengaps) + 1
    optimal_k_base = max(min_clusters, min(optimal_k_base, max_clusters))

    # Apply oversampling
    optimal_k = int(np.round(optimal_k_base * oversample_factor))
    optimal_k = max(min_clusters, min(optimal_k, max_clusters))

    logger.info(
        "Estimated optimal number of clusters: base=%d, oversampled=%d (factor=%.2f)",
        optimal_k_base, optimal_k, oversample_factor
    )

    return optimal_k


def merge_small_clusters(
    cluster_assignments: np.ndarray,
    building_ids: List,
    voronoi_gdf: gpd.GeoDataFrame,
    gat_labels: np.ndarray,
    cluster_to_label: Dict[int, int],
    area_threshold_m2: float = 1_000_000
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Merge clusters whose total voronoi area is below threshold with neighboring clusters.

    Args:
        cluster_assignments: Current cluster assignments (N,)
        building_ids: List of building IDs corresponding to cluster assignments
        voronoi_gdf: GeoDataFrame containing voronoi polygons with building IDs
        gat_labels: GAT predicted labels (N,)
        cluster_to_label: Current mapping from cluster ID to GAT label
        area_threshold_m2: Area threshold in square meters (default: 1km² = 1,000,000 m²)

    Returns:
        merged_assignments: Updated cluster assignments after merging (N,)
        merged_cluster_to_label: Updated cluster-to-label mapping
    """
    if voronoi_gdf is None or len(voronoi_gdf) == 0:
        logger.warning("No voronoi data provided, skipping cluster merging")
        return cluster_assignments, cluster_to_label

    logger.info("Starting cluster merging with area threshold %.2f km²", area_threshold_m2 / 1_000_000)

    # Create mapping from building ID to cluster
    building_to_cluster = {bid: cluster for bid, cluster in zip(building_ids, cluster_assignments)}
    building_to_gat_label = {bid: label for bid, label in zip(building_ids, gat_labels)}

    # Find ID field in voronoi GeoDataFrame
    id_field = None
    for possible_id in ['FID', 'OBJECTID', 'ID', 'id', 'building_id']:
        if possible_id in voronoi_gdf.columns:
            id_field = possible_id
            break

    if id_field is None:
        logger.warning("No ID field found in voronoi data, cannot merge clusters")
        return cluster_assignments, cluster_to_label

    # Assign cluster ID to each voronoi polygon
    voronoi_gdf = voronoi_gdf.copy()
    voronoi_gdf['cluster'] = voronoi_gdf[id_field].map(building_to_cluster)
    voronoi_gdf = voronoi_gdf[voronoi_gdf['cluster'].notna()].copy()

    # Calculate total area per cluster
    cluster_areas = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster_id in unique_clusters:
        cluster_voronoi = voronoi_gdf[voronoi_gdf['cluster'] == cluster_id]
        if len(cluster_voronoi) > 0:
            total_area = cluster_voronoi.geometry.area.sum()
            cluster_areas[int(cluster_id)] = total_area
        else:
            cluster_areas[int(cluster_id)] = 0.0

    logger.debug("Cluster areas (m²): %s", {k: f"{v:.2f}" for k, v in cluster_areas.items()})

    # Identify small clusters
    small_clusters = [cid for cid, area in cluster_areas.items() if area < area_threshold_m2]
    logger.info("Found %d clusters below threshold (%.2f km²): %s",
                len(small_clusters), area_threshold_m2 / 1_000_000, small_clusters)

    if not small_clusters:
        logger.info("No small clusters to merge")
        return cluster_assignments, cluster_to_label

    # Create adjacency graph of clusters (which clusters are neighbors)
    cluster_neighbors = {int(cid): set() for cid in unique_clusters}

    # Build cluster adjacency from voronoi topology
    for idx1 in range(len(voronoi_gdf)):
        for idx2 in range(idx1 + 1, len(voronoi_gdf)):
            geom1 = voronoi_gdf.iloc[idx1].geometry
            geom2 = voronoi_gdf.iloc[idx2].geometry
            cluster1 = int(voronoi_gdf.iloc[idx1]['cluster'])
            cluster2 = int(voronoi_gdf.iloc[idx2]['cluster'])

            if cluster1 != cluster2 and geom1.touches(geom2):
                cluster_neighbors[cluster1].add(cluster2)
                cluster_neighbors[cluster2].add(cluster1)

    # Merge small clusters with neighboring clusters of same GAT label
    merged_assignments = cluster_assignments.copy()
    merge_mapping = {}  # Map old cluster ID to new cluster ID

    for small_cluster in sorted(small_clusters):
        # Get GAT label for this cluster
        small_cluster_label = cluster_to_label.get(small_cluster)

        # Find neighboring clusters with same GAT label
        neighbors = cluster_neighbors.get(small_cluster, set())
        same_label_neighbors = [
            n for n in neighbors
            if cluster_to_label.get(n) == small_cluster_label and n not in small_clusters
        ]

        if same_label_neighbors:
            # Merge with the largest neighboring cluster with same label
            target_cluster = max(same_label_neighbors, key=lambda c: cluster_areas.get(c, 0))
            merge_mapping[small_cluster] = target_cluster

            logger.info(
                "Merging cluster %d (area=%.2f km², label=%d) into cluster %d (area=%.2f km²)",
                small_cluster, cluster_areas.get(small_cluster, 0) / 1_000_000,
                small_cluster_label, target_cluster, cluster_areas.get(target_cluster, 0) / 1_000_000
            )
        else:
            logger.warning(
                "Cluster %d (area=%.2f km², label=%d) has no suitable neighbors for merging",
                small_cluster, cluster_areas.get(small_cluster, 0) / 1_000_000, small_cluster_label
            )

    # Apply merging
    for old_cluster, new_cluster in merge_mapping.items():
        merged_assignments[cluster_assignments == old_cluster] = new_cluster

    # Update cluster_to_label mapping (remove merged clusters)
    merged_cluster_to_label = {k: v for k, v in cluster_to_label.items() if k not in merge_mapping}

    logger.info("Cluster merging completed: %d clusters merged, %d clusters remaining",
                len(merge_mapping), len(np.unique(merged_assignments)))

    return merged_assignments, merged_cluster_to_label


def perform_spectral_clustering_pipeline(
    embeddings: np.ndarray,
    features: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    gat_labels: np.ndarray,
    building_ids: Optional[List] = None,
    voronoi_gdf: Optional[gpd.GeoDataFrame] = None,
    n_clusters: Optional[int] = None,
    embedding_weight: float = 0.5,
    feature_weight: float = 0.3,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0,
    area_threshold_m2: float = 1_000_000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Complete spectral clustering pipeline with oversampling and area-based merging.

    Args:
        embeddings: GAT embeddings (N, D_emb)
        features: Clustering features (N, D_feat)
        adjacency_matrix: Distance-based adjacency (N, N)
        gat_labels: GAT predicted labels (N,)
        building_ids: List of building IDs (for voronoi matching)
        voronoi_gdf: GeoDataFrame with voronoi polygons (for area-based merging)
        n_clusters: Number of clusters (auto-estimate with 1.5x oversampling if None)
        embedding_weight: Weight for embedding similarity
        feature_weight: Weight for feature similarity
        distance_weight: Weight for distance-based adjacency
        distance_scale: Scale factor for distance conversion
        area_threshold_m2: Minimum cluster area threshold (default: 1 km²)
        random_state: Random seed

    Returns:
        cluster_assignments: Final cluster assignments after merging (N,)
        final_labels: Final labels after assigning GAT labels (N,)
        cluster_to_label: Mapping from cluster to GAT label
        affinity_matrix: Computed affinity matrix (N, N)
    """
    logger.info("Starting spectral clustering pipeline")

    # Step 1: Compute affinity matrix
    affinity_matrix = compute_affinity_matrix(
        embeddings=embeddings,
        features=features,
        adjacency_matrix=adjacency_matrix,
        embedding_weight=embedding_weight,
        feature_weight=feature_weight,
        distance_weight=distance_weight,
        distance_scale=distance_scale
    )

    # Step 2: Estimate optimal clusters if not provided (with 1.5x oversampling)
    if n_clusters is None:
        n_clusters = estimate_optimal_clusters(affinity_matrix, oversample_factor=1.5)

    # Step 3: Perform spectral clustering
    cluster_assignments = spectral_cluster(
        affinity_matrix=affinity_matrix,
        n_clusters=n_clusters,
        random_state=random_state
    )

    # Step 4: Assign GAT labels to clusters
    cluster_to_label, final_labels = assign_labels_to_clusters(
        cluster_assignments=cluster_assignments,
        gat_labels=gat_labels
    )

    # Step 5: Merge small clusters based on voronoi area threshold
    if building_ids is not None and voronoi_gdf is not None:
        cluster_assignments, cluster_to_label = merge_small_clusters(
            cluster_assignments=cluster_assignments,
            building_ids=building_ids,
            voronoi_gdf=voronoi_gdf,
            gat_labels=gat_labels,
            cluster_to_label=cluster_to_label,
            area_threshold_m2=area_threshold_m2
        )

        # Update final labels after merging
        final_labels = np.array([cluster_to_label[int(c)] for c in cluster_assignments])
        logger.info("Updated final labels after cluster merging")
    else:
        logger.info("Skipping cluster merging (no building IDs or voronoi data provided)")

    logger.info("Spectral clustering pipeline completed successfully")

    return cluster_assignments, final_labels, cluster_to_label, affinity_matrix

