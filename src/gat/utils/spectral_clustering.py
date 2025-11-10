"""Spectral clustering utilities for building clustering.

This module implements a two-stage classification + spatial smoothing approach:

1. Stage 1 (GAT): Classification using discriminative features
   - Predicts building labels based on learned representations
   - Provides initial classification and confidence scores

2. Stage 2 (Spectral Clustering): Spatial smoothing using morphological features
   - Groups buildings based on morphological similarity and spatial proximity
   - Uses majority voting (optionally confidence-weighted) to assign labels to clusters
   - Ensures spatial consistency: nearby similar buildings get the same label

Design Rationale:
- GAT focuses on "what type?" (discriminative task)
- Spectral clustering focuses on "which belong together?" (grouping task)
- Two feature sets for two objectives: better task-specific performance
- Confidence weighting: trust high-confidence GAT predictions more
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

from .logger import get_logger

logger = get_logger()


def compute_confidence(logits: np.ndarray) -> np.ndarray:
    """
    Compute prediction confidence from GAT logits.

    Confidence is defined as the maximum softmax probability across all classes.
    Higher confidence indicates that GAT is more certain about its prediction.

    Args:
        logits: GAT classification logits, shape (N, num_classes)

    Returns:
        confidence: Confidence scores for each prediction, shape (N,)
                   Values range from 0 to 1, where 1 means completely certain

    Example:
        logits = [[2.0, 1.0, 0.5]]  # GAT strongly prefers class 0
        confidence = compute_confidence(logits)  # Returns ~0.66 (high confidence)
    """
    from scipy.special import softmax

    # Convert logits to probabilities
    probs = softmax(logits, axis=1)

    # Confidence is the maximum probability
    confidence = np.max(probs, axis=1)

    return confidence


def compute_affinity_matrix(
    embeddings: np.ndarray,
    features: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    embedding_weight: float = 0.3,
    feature_weight: float = 0.5,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0
) -> np.ndarray:
    """
    Compute affinity matrix for spectral clustering by combining multiple similarity measures.

    This function implements the core of the spatial smoothing strategy:
    - Embedding similarity: Captures GAT's learned discriminative features
    - Feature similarity: Captures morphological similarity for grouping
    - Distance affinity: Enforces spatial proximity constraint

    Weight Rationale (default: 0.3/0.5/0.2):
    - Lower embedding_weight (0.3): GAT already performed classification, avoid over-reliance
    - Higher feature_weight (0.5): Focus on morphological similarity for spatial grouping
    - Distance_weight (0.2): Spatial constraint to prevent distant buildings from clustering

    Args:
        embeddings: GAT node embeddings (N, D_emb) - discriminative features learned by GAT
        features: Morphological building features (N, D_feat) - shape, size, orientation, etc.
        adjacency_matrix: Distance-based adjacency matrix (N, N) - spatial relationships
        embedding_weight: Weight for embedding similarity (default 0.3)
        feature_weight: Weight for feature similarity (default 0.5)
        distance_weight: Weight for distance-based adjacency (default 0.2)
        distance_scale: Scale factor for distance-to-affinity conversion in meters (default 1.0)

    Returns:
        affinity_matrix: Combined affinity matrix (N, N), normalized to [0, 1]

    Note:
        Weights should sum to 1.0 for proper normalization.
        Adjust weights based on validation performance.
    """
    logger.debug(
        "Computing affinity matrix: embeddings=%s, features=%s, adjacency=%s, weights=[%.2f,%.2f,%.2f]",
        embeddings.shape, features.shape, adjacency_matrix.shape,
        embedding_weight, feature_weight, distance_weight
    )

    # 1. Embedding-based similarity (captures discriminative information from GAT)
    # Using cosine similarity: measures angle between embedding vectors
    embedding_sim = cosine_similarity(embeddings)
    # Normalize from [-1, 1] to [0, 1] range
    embedding_sim = (embedding_sim + 1) / 2
    logger.debug("Embedding similarity computed: min=%.4f, max=%.4f, mean=%.4f", 
                 embedding_sim.min(), embedding_sim.max(), embedding_sim.mean())

    # 2. Feature-based similarity (captures morphological similarity for grouping)
    # Using cosine similarity on morphological features (area, shape, orientation, etc.)
    feature_sim = cosine_similarity(features)
    # Normalize from [-1, 1] to [0, 1] range
    feature_sim = (feature_sim + 1) / 2
    logger.debug("Feature similarity computed: min=%.4f, max=%.4f, mean=%.4f", 
                 feature_sim.min(), feature_sim.max(), feature_sim.mean())

    # 3. Distance-based affinity (enforces spatial proximity constraint)
    # Adjacency matrix contains distances between buildings (meters)
    # Convert distance to affinity using exponential decay: affinity = exp(-distance / scale)
    # Closer buildings have higher affinity, distant buildings have lower affinity
    distance_matrix = adjacency_matrix.values

    # Apply exponential decay to convert distances to affinities
    # distance_scale controls how quickly affinity decays with distance
    distance_affinity = np.exp(-distance_matrix / distance_scale)

    # Set diagonal to 1 (self-similarity is maximum)
    np.fill_diagonal(distance_affinity, 1.0)

    # Zero out non-adjacent nodes (where original adjacency was 0 = no connection)
    distance_affinity[distance_matrix == 0] = 0

    logger.debug("Distance affinity computed: min=%.4f, max=%.4f, mean=%.4f (non-zero)", 
                 distance_affinity[distance_affinity > 0].min(), 
                 distance_affinity.max(),
                 distance_affinity[distance_affinity > 0].mean())

    # 4. Combine all three similarity measures using weighted sum
    # This is the core of the two-stage approach:
    # - Embeddings provide discriminative information (what GAT learned)
    # - Features provide morphological similarity (for spatial grouping)
    # - Distance provides spatial constraint (only nearby buildings cluster together)
    affinity = (
        embedding_weight * embedding_sim +
        feature_weight * feature_sim +
        distance_weight * distance_affinity
    )

    logger.debug("Combined affinity contributions: emb=%.4f, feat=%.4f, dist=%.4f",
                 (embedding_weight * embedding_sim).mean(),
                 (feature_weight * feature_sim).mean(),
                 (distance_weight * distance_affinity[distance_affinity > 0]).mean())

    # Normalize combined affinity to [0, 1] range for stability
    affinity_min = affinity.min()
    affinity_max = affinity.max()
    if affinity_max > affinity_min:
        affinity = (affinity - affinity_min) / (affinity_max - affinity_min)

    # Ensure symmetry (required for spectral clustering)
    affinity = (affinity + affinity.T) / 2

    # Set diagonal to 1 (maximum self-affinity)
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
) -> Tuple[dict, np.ndarray]:
    """
    Assign GAT predicted labels to spectral clusters using simple majority voting.

    This is the basic voting strategy: each building gets one vote, and the most
    common label in each cluster wins.

    Args:
        cluster_assignments: Spectral cluster assignments (N,)
        gat_labels: GAT predicted labels (N,)

    Returns:
        cluster_to_label: Mapping from cluster ID to GAT label (dict)
        final_labels: Final label for each node based on cluster assignment (N,)
    """
    logger.debug("Assigning GAT labels to spectral clusters via simple majority voting")

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

    logger.info("Assigned GAT labels to %d clusters using simple majority voting", len(cluster_to_label))

    return cluster_to_label, final_labels


def assign_labels_with_confidence(
    cluster_assignments: np.ndarray,
    gat_labels: np.ndarray,
    gat_logits: np.ndarray
) -> Tuple[dict, np.ndarray]:
    """
    Assign GAT labels to clusters using confidence-weighted majority voting.

    Unlike simple majority voting where each building gets one vote, this method
    weights each vote by GAT's prediction confidence. Buildings where GAT is very
    confident have more influence on the cluster's final label.

    Benefits:
    - Prevents low-confidence wrong predictions from being "voted up"
    - Trusts high-confidence GAT predictions more
    - More robust at cluster boundaries where GAT may be uncertain

    Example:
        Cluster with 3 buildings:
        - Building A: Label 0, confidence 0.9 → weighted vote = 0.9 for Label 0
        - Building B: Label 1, confidence 0.4 → weighted vote = 0.4 for Label 1  
        - Building C: Label 1, confidence 0.3 → weighted vote = 0.3 for Label 1
        Result: Label 0 wins (0.9 > 0.7), even though Label 1 has more votes

    Args:
        cluster_assignments: Spectral cluster assignments (N,)
        gat_labels: GAT predicted labels (N,)
        gat_logits: GAT classification logits (N, num_classes)

    Returns:
        cluster_to_label: Mapping from cluster ID to GAT label (dict)
        final_labels: Final label for each node based on cluster assignment (N,)
    """
    logger.debug("Assigning GAT labels to spectral clusters via confidence-weighted majority voting")

    # Compute confidence scores from logits
    confidence = compute_confidence(gat_logits)

    logger.debug("Confidence statistics: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
                 confidence.min(), confidence.max(), confidence.mean(), np.median(confidence))

    unique_clusters = np.unique(cluster_assignments)
    cluster_to_label = {}

    for cluster_id in unique_clusters:
        # Find all nodes in this cluster
        mask = cluster_assignments == cluster_id
        cluster_labels = gat_labels[mask]
        cluster_confidence = confidence[mask]

        # Confidence-weighted voting: sum confidence scores for each label
        label_weights = {}
        for label, conf in zip(cluster_labels, cluster_confidence):
            label = int(label)
            label_weights[label] = label_weights.get(label, 0.0) + conf

        # Select label with highest weighted sum
        majority_label = max(label_weights, key=label_weights.get)
        cluster_to_label[int(cluster_id)] = majority_label

        # Count simple votes for comparison
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        vote_counts = dict(zip(unique_labels.tolist(), counts.tolist()))

        logger.debug(
            "Cluster %d: %d nodes, simple votes=%s, weighted votes=%s, assigned label=%d",
            cluster_id, mask.sum(), vote_counts, 
            {k: f"{v:.2f}" for k, v in label_weights.items()}, 
            majority_label
        )

    # Create final labels array
    final_labels = np.array([cluster_to_label[int(c)] for c in cluster_assignments])

    logger.info("Assigned GAT labels to %d clusters using confidence-weighted voting", 
                len(cluster_to_label))

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
    gat_logits: Optional[np.ndarray] = None,
    use_confidence_weighted_voting: bool = True,
    embedding_weight: float = 0.3,
    feature_weight: float = 0.5,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0,
    area_threshold_m2: float = 1_000_000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Complete spectral clustering pipeline for spatial smoothing of GAT predictions.

    This pipeline implements the second stage of our two-stage approach:
    1. Compute affinity matrix combining embeddings, morphological features, and spatial distance
    2. Perform spectral clustering to group spatially similar buildings
    3. Assign GAT labels to clusters using (optionally confidence-weighted) majority voting
    4. Merge small clusters based on area threshold

    The key innovation is the confidence-weighted voting: GAT predictions with high
    confidence have more influence, preventing uncertain predictions from dominating.

    Args:
        embeddings: GAT embeddings (N, D_emb) - discriminative features from GAT
        features: Morphological clustering features (N, D_feat) - shape, size, orientation
        adjacency_matrix: Distance-based adjacency (N, N) - spatial relationships
        gat_labels: GAT predicted labels (N,) - initial classification from GAT
        building_ids: List of building IDs (for voronoi matching in merging)
        voronoi_gdf: GeoDataFrame with voronoi polygons (for area-based merging)
        n_clusters: Number of clusters (auto-estimate with oversampling if None)
        gat_logits: GAT classification logits (N, num_classes) - for confidence weighting
        use_confidence_weighted_voting: Whether to use confidence-weighted voting (default True)
        embedding_weight: Weight for embedding similarity (default 0.3)
        feature_weight: Weight for morphological feature similarity (default 0.5)
        distance_weight: Weight for distance-based adjacency (default 0.2)
        distance_scale: Scale factor for distance-to-affinity conversion in meters (default 1.0)
        area_threshold_m2: Minimum cluster area threshold in square meters (default 1 km²)
        random_state: Random seed for reproducibility

    Returns:
        cluster_assignments: Final cluster assignments after merging (N,)
        final_labels: Final labels after assigning GAT labels (N,)
        cluster_to_label: Mapping from cluster ID to GAT label (dict)
        affinity_matrix: Computed affinity matrix (N, N)
    """
    logger.info("Starting spectral clustering pipeline (weights: emb=%.2f, feat=%.2f, dist=%.2f, conf_weighted=%s)",
                embedding_weight, feature_weight, distance_weight, use_confidence_weighted_voting)

    # Step 1: Compute affinity matrix combining embeddings, features, and spatial distance
    affinity_matrix = compute_affinity_matrix(
        embeddings=embeddings,
        features=features,
        adjacency_matrix=adjacency_matrix,
        embedding_weight=embedding_weight,
        feature_weight=feature_weight,
        distance_weight=distance_weight,
        distance_scale=distance_scale
    )

    # Step 2: Estimate optimal number of clusters if not provided (with 1.5x oversampling)
    # Oversampling helps capture fine-grained spatial structure, small clusters merged later
    if n_clusters is None:
        n_clusters = estimate_optimal_clusters(affinity_matrix, oversample_factor=1.5)

    # Step 3: Perform spectral clustering to group spatially similar buildings
    cluster_assignments = spectral_cluster(
        affinity_matrix=affinity_matrix,
        n_clusters=n_clusters,
        random_state=random_state
    )

    # Step 4: Assign GAT labels to clusters using majority voting
    # Choose voting strategy based on availability of logits and configuration
    if use_confidence_weighted_voting and gat_logits is not None:
        logger.info("Using confidence-weighted majority voting")
        cluster_to_label, final_labels = assign_labels_with_confidence(
            cluster_assignments=cluster_assignments,
            gat_labels=gat_labels,
            gat_logits=gat_logits
        )
    else:
        if use_confidence_weighted_voting and gat_logits is None:
            logger.warning("Confidence-weighted voting requested but logits not provided, "
                          "falling back to simple majority voting")
        logger.info("Using simple majority voting")
        cluster_to_label, final_labels = assign_labels_to_clusters(
            cluster_assignments=cluster_assignments,
            gat_labels=gat_labels
        )

    # Step 5: Merge small clusters based on voronoi area threshold
    # This ensures spatially significant clusters while maintaining spatial consistency
    if building_ids is not None and voronoi_gdf is not None:
        logger.info("Merging small clusters (threshold: %.2f km²)", area_threshold_m2 / 1_000_000)
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

    logger.info("Spectral clustering pipeline completed successfully: %d final clusters",
                len(np.unique(cluster_assignments)))

    return cluster_assignments, final_labels, cluster_to_label, affinity_matrix

