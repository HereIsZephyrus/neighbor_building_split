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
from typing import Tuple, Optional, List, Dict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

from .logger import get_logger

logger = get_logger(__name__)


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
    distance_scale: float = 1.0,
    max_hops: int = 3
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
        max_hops: Maximum graph hops for clustering (default 3)

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

    # Get distance matrix first (needed for masking non-adjacent nodes)
    distance_matrix = adjacency_matrix.values

    # Create adjacency mask (True for adjacent nodes, False for non-adjacent)
    adjacency_mask = distance_matrix > 0
    np.fill_diagonal(adjacency_mask, True)  # Self-connections always allowed

    # 1. Embedding-based similarity (captures discriminative information from GAT)
    # Using cosine similarity: measures angle between embedding vectors
    embedding_sim = cosine_similarity(embeddings)
    # Normalize from [-1, 1] to [0, 1] range
    embedding_sim = (embedding_sim + 1) / 2

    # CRITICAL: Restrict embedding similarity to adjacent nodes only
    # This prevents distant buildings from clustering based on embedding similarity alone
    embedding_sim[~adjacency_mask] = 0

    logger.debug("Embedding similarity computed: min=%.4f, max=%.4f, mean=%.4f (non-zero)", 
                 embedding_sim[embedding_sim > 0].min() if embedding_sim[embedding_sim > 0].size > 0 else 0,
                 embedding_sim.max(), 
                 embedding_sim[embedding_sim > 0].mean() if embedding_sim[embedding_sim > 0].size > 0 else 0)

    # 2. Feature-based similarity (captures morphological similarity for grouping)
    # Using cosine similarity on morphological features (area, shape, orientation, etc.)
    feature_sim = cosine_similarity(features)
    # Normalize from [-1, 1] to [0, 1] range
    feature_sim = (feature_sim + 1) / 2

    # CRITICAL: Restrict feature similarity to adjacent nodes only
    # This prevents distant buildings from clustering based on morphological similarity alone
    feature_sim[~adjacency_mask] = 0

    logger.debug("Feature similarity computed: min=%.4f, max=%.4f, mean=%.4f (non-zero)", 
                 feature_sim[feature_sim > 0].min() if feature_sim[feature_sim > 0].size > 0 else 0,
                 feature_sim.max(), 
                 feature_sim[feature_sim > 0].mean() if feature_sim[feature_sim > 0].size > 0 else 0)

    # 3. Distance-based affinity (enforces spatial proximity constraint)
    # Adjacency matrix contains distances between buildings (meters)
    # Convert distance to affinity using exponential decay: affinity = exp(-distance / scale)
    # Closer buildings have higher affinity, distant buildings have lower affinity

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

    logger.info("Combined affinity contributions: emb=%.4f, feat=%.4f, dist=%.4f",
                 (embedding_weight * embedding_sim).mean(),
                 (feature_weight * feature_sim).mean(),
                 (distance_weight * distance_affinity[distance_affinity > 0]).mean())

    # Ensure symmetry (required for spectral clustering)
    affinity = (affinity + affinity.T) / 2

    # Set diagonal to 1 (maximum self-affinity)
    np.fill_diagonal(affinity, 1.0)

    logger.debug("Final affinity matrix: min=%.4f, max=%.4f, mean=%.4f", 
                 affinity.min(), affinity.max(), affinity.mean())

    # Apply hop constraint to prevent distant buildings from clustering
    if max_hops is not None and max_hops > 0:
        affinity = apply_hop_constraint(affinity, adjacency_matrix, max_hops)

    return affinity


def apply_hop_constraint(
    affinity_matrix: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    max_hops: int = 3
) -> np.ndarray:
    """
    Apply maximum hop constraint to affinity matrix.

    Limits clustering propagation by zeroing out affinities between nodes
    that are more than max_hops apart in the graph topology.

    Args:
        affinity_matrix: Computed affinity matrix (N, N)
        adjacency_matrix: Original adjacency matrix (N, N) 
        max_hops: Maximum number of hops allowed (default 3)

    Returns:
        constrained_affinity: Affinity matrix with hop constraint applied (N, N)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    logger.debug(f"Applying {max_hops}-hop constraint to affinity matrix")

    # Create binary adjacency for hop distance calculation
    adjacency_binary = (adjacency_matrix.values > 0).astype(float)
    sparse_adj = csr_matrix(adjacency_binary)

    # Compute shortest path distances (in hops)
    hop_distances = shortest_path(
        sparse_adj, 
        directed=False, 
        return_predecessors=False,
        unweighted=True
    )

    # Create hop mask: True for nodes within max_hops
    hop_mask = (hop_distances <= max_hops) & np.isfinite(hop_distances)
    np.fill_diagonal(hop_mask, True)

    # Apply constraint
    constrained_affinity = affinity_matrix * hop_mask

    num_original = (affinity_matrix > 0).sum()
    num_constrained = (constrained_affinity > 0).sum()

    logger.info(
        f"Hop constraint applied: reduced non-zero entries from {num_original} to {num_constrained} "
        f"({100.0 * num_constrained / num_original:.1f}%)"
    )

    return constrained_affinity


def filter_small_clusters(
    cluster_assignments: np.ndarray,
    cluster_to_label: dict,
    gat_labels: np.ndarray,
    min_cluster_size: int = 5
) -> Tuple[dict, np.ndarray, dict, np.ndarray]:
    """
    Filter out clusters smaller than threshold and revert to GAT predictions.

    For clusters with size < min_cluster_size:
    - Do not use cluster majority voting result
    - Revert buildings to original GAT predicted labels
    - Mark these buildings with negative cluster IDs (for visualization exclusion)

    Args:
        cluster_assignments: Original cluster assignments (N,)
        cluster_to_label: Cluster ID to label mapping (dict)
        gat_labels: Original GAT predicted labels (N,)
        min_cluster_size: Minimum cluster size threshold (default 5)

    Returns:
        filtered_cluster_to_label: Filtered cluster to label mapping (only valid clusters)
        final_labels: Final labels with small clusters reverted to GAT predictions (N,)
        cluster_stats: Statistics about filtering operation (dict)
        cleaned_cluster_assignments: Cluster assignments with small clusters marked as negative IDs (N,)
    """
    logger.debug(f"Filtering clusters with size < {min_cluster_size}")

    # Count cluster sizes
    unique_clusters, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
    cluster_size_dict = dict(zip(unique_clusters, cluster_sizes))

    # Identify small and valid clusters
    small_clusters = []
    valid_clusters = []

    for cluster_id, size in cluster_size_dict.items():
        if size < min_cluster_size:
            small_clusters.append(cluster_id)
        else:
            valid_clusters.append(cluster_id)

    logger.debug(
        f"Cluster filtering: {len(small_clusters)} small clusters (< {min_cluster_size}), "
        f"{len(valid_clusters)} valid clusters"
    )

    # Filter cluster_to_label (keep only valid clusters)
    filtered_cluster_to_label = {
        cid: cluster_to_label[cid] 
        for cid in valid_clusters
    }

    # Build final labels array and cleaned cluster assignments
    final_labels = np.zeros_like(gat_labels)
    cleaned_cluster_assignments = cluster_assignments.copy()
    revert_count = 0

    # Assign negative IDs to small/filtered clusters to exclude them from visualization
    next_invalid_id = -1

    for i in range(len(cluster_assignments)):
        cluster_id = cluster_assignments[i]

        if cluster_id in small_clusters:
            # Small cluster: revert to GAT prediction
            final_labels[i] = gat_labels[i]
            # Mark with negative ID for visualization exclusion
            cleaned_cluster_assignments[i] = next_invalid_id
            revert_count += 1
        else:
            # Valid cluster: use majority voting result
            final_labels[i] = cluster_to_label[cluster_id]

    # Update invalid ID counter for next small cluster
    if revert_count > 0:
        next_invalid_id -= 1

    logger.debug(f"Reverted {revert_count} buildings from small clusters to GAT predictions")

    # Build statistics
    cluster_stats = {
        'total_clusters': len(unique_clusters),
        'valid_clusters': len(valid_clusters),
        'small_clusters': len(small_clusters),
        'small_cluster_ids': small_clusters,
        'buildings_reverted': revert_count,
        'cluster_sizes': cluster_size_dict
    }

    # Detailed logging for small clusters
    if small_clusters:
        logger.debug("Small clusters detail:")
        for cid in small_clusters:
            size = cluster_size_dict[cid]
            original_label = cluster_to_label.get(cid, 'N/A')
            logger.debug(f"  Cluster {cid}: size={size}, original_label={original_label}")

    return filtered_cluster_to_label, final_labels, cluster_stats, cleaned_cluster_assignments


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

    logger.debug("Assigned GAT labels to %d clusters using simple majority voting", len(cluster_to_label))

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

    logger.debug("Assigned GAT labels to %d clusters using confidence-weighted voting", 
                len(cluster_to_label))

    return cluster_to_label, final_labels


def estimate_optimal_clusters(
    affinity_matrix: np.ndarray,
    max_clusters: int = 15,
    min_clusters: int = 2,
    oversample_factor: float = 1,
    num_buildings: Optional[int] = None,
    min_cluster_size: Optional[int] = None
) -> int:
    """
    Estimate optimal number of clusters using eigenvalue analysis with oversampling.

    Args:
        affinity_matrix: Affinity matrix (N, N)
        max_clusters: Maximum number of clusters to consider (default 15)
        min_clusters: Minimum number of clusters (default 2)
        oversample_factor: Multiplier for oversampling clusters (default 1.5)
        num_buildings: Number of buildings in component (for dynamic max calculation)
        min_cluster_size: Minimum cluster size (for dynamic max calculation)

    Returns:
        optimal_k: Estimated optimal number of clusters (with oversampling)

    Note:
        If num_buildings and min_cluster_size are provided, the maximum number of clusters
        is constrained to max(1, num_buildings / (min_cluster_size * 2)) to prevent
        over-clustering and ensure clusters are meaningful.
    """
    from scipy.linalg import eigh

    logger.debug("Estimating optimal number of clusters via eigenvalue analysis")

    # Calculate dynamic max clusters based on component size and min_cluster_size
    if num_buildings is not None and min_cluster_size is not None and min_cluster_size > 0:
        # Constraint: max clusters = max(1, n / (min_cluster_size * 2))
        # This ensures each cluster can reasonably contain min_cluster_size buildings
        dynamic_max = max(1, int(num_buildings / (min_cluster_size * 2)))
        max_clusters_limit = min(dynamic_max, 10, len(affinity_matrix) - 1)
        logger.debug(
            f"Dynamic max clusters: {dynamic_max} (n={num_buildings}, min_size={min_cluster_size}), "
            f"final limit={max_clusters_limit}"
        )
    else:
        # Fallback to fixed limit
        max_clusters_limit = min(10, len(affinity_matrix) - 1)
        logger.debug(f"Using fixed max clusters limit: {max_clusters_limit}")

    # Compute Laplacian
    degree_matrix = np.diag(affinity_matrix.sum(axis=1))
    laplacian = degree_matrix - affinity_matrix

    # Compute eigenvalues
    eigenvalues, _ = eigh(laplacian)

    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)

    # Find eigengap (largest gap in first max_clusters eigenvalues)
    eigengaps = np.diff(eigenvalues[:max_clusters_limit + 1])

    optimal_k_base = np.argmax(eigengaps) + 1
    # Force minimum K = 1, maximum K = max_clusters_limit
    optimal_k_base = max(1, min(optimal_k_base, max_clusters_limit))

    # Apply oversampling
    optimal_k = int(np.round(optimal_k_base * oversample_factor))
    # Force minimum K = 1, maximum K = max_clusters_limit
    optimal_k = max(1, min(optimal_k, max_clusters_limit))

    logger.debug(
        "Estimated optimal number of clusters: base=%d, oversampled=%d (factor=%.2f, max_limit=%d)",
        optimal_k_base, optimal_k, oversample_factor, max_clusters_limit
    )

    return optimal_k


def perform_spectral_clustering_pipeline(
    embeddings: np.ndarray,
    features: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    gat_labels: np.ndarray,
    building_ids: Optional[List] = None,
    voronoi_areas: Optional[Dict[int, float]] = None,
    n_clusters: Optional[int] = None,
    gat_logits: Optional[np.ndarray] = None,
    use_confidence_weighted_voting: bool = True,
    embedding_weight: float = 0.3,
    feature_weight: float = 0.5,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0,
    min_cluster_size: int = 5,
    max_hops: int = 3,
    oversample_factor: float = 1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, dict]:
    """
    Complete spectral clustering pipeline for spatial smoothing of GAT predictions.

    This pipeline implements the second stage of our two-stage approach:
    1. Compute affinity matrix combining embeddings, morphological features, and spatial distance
    2. Perform spectral clustering to group spatially similar buildings
    3. Assign GAT labels to clusters using (optionally confidence-weighted) majority voting
    4. Filter small clusters and revert to GAT predictions

    The key innovation is the confidence-weighted voting: GAT predictions with high
    confidence have more influence, preventing uncertain predictions from dominating.

    Args:
        embeddings: GAT embeddings (N, D_emb) - discriminative features from GAT
        features: Morphological clustering features (N, D_feat) - shape, size, orientation
        adjacency_matrix: Distance-based adjacency (N, N) - spatial relationships
        gat_labels: GAT predicted labels (N,) - initial classification from GAT
        building_ids: List of building IDs (optional, for logging)
        voronoi_areas: Dictionary mapping building ID to voronoi area in m² (optional, for logging)
        n_clusters: Number of clusters (auto-estimate with oversampling if None)
        gat_logits: GAT classification logits (N, num_classes) - for confidence weighting
        use_confidence_weighted_voting: Whether to use confidence-weighted voting (default True)
        embedding_weight: Weight for embedding similarity (default 0.3)
        feature_weight: Weight for morphological feature similarity (default 0.5)
        distance_weight: Weight for distance-based adjacency (default 0.2)
        distance_scale: Scale factor for distance-to-affinity conversion in meters (default 1.0)
        min_cluster_size: Minimum buildings per cluster; smaller reverted to GAT (default 5)
        max_hops: Maximum graph hops for clustering (default 3)
        oversample_factor: Multiplier for oversampling clusters (default 1.5)
        random_state: Random seed for reproducibility

    Returns:
        cluster_assignments: Final cluster assignments (N,)
        final_labels: Final labels after filtering small clusters (N,)
        cluster_to_label: Mapping from cluster ID to label (only valid clusters)
        affinity_matrix: Computed affinity matrix (N, N)
        cluster_stats: Statistics about cluster filtering (dict)
    """
    logger.info(
        "Starting spectral clustering pipeline "
        "(weights: emb=%.2f, feat=%.2f, dist=%.2f, conf_weighted=%s, min_size=%d, max_hops=%d)",
        embedding_weight, feature_weight, distance_weight, 
        use_confidence_weighted_voting, min_cluster_size, max_hops
    )

    # Step 1: Compute affinity matrix combining embeddings, features, and spatial distance
    # This includes hop constraint which may disconnect the graph
    affinity_matrix = compute_affinity_matrix(
        embeddings=embeddings,
        features=features,
        adjacency_matrix=adjacency_matrix,
        embedding_weight=embedding_weight,
        feature_weight=feature_weight,
        distance_weight=distance_weight,
        distance_scale=distance_scale,
        max_hops=max_hops
    )

    # Step 2: CRITICAL - Identify sub-connected components after hop constraint
    # The hop constraint may have disconnected the original component into smaller pieces
    # We need to cluster each piece independently to ensure spatial contiguity
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as scipy_connected_components

    # Create binary affinity matrix (1 if connected, 0 otherwise)
    affinity_binary = (affinity_matrix > 0).astype(int)
    sparse_affinity = csr_matrix(affinity_binary)

    # Find sub-connected components
    num_sub_components, sub_component_labels = scipy_connected_components(
        csgraph=sparse_affinity,
        directed=False,
        return_labels=True
    )

    logger.debug(
        f"After hop constraint, found {num_sub_components} sub-connected components "
        f"(original component split by max_hops={max_hops})"
    )

    # Step 3: Process each sub-component independently
    all_cluster_assignments = np.zeros(len(embeddings), dtype=int)
    all_final_labels = np.zeros(len(embeddings), dtype=int)
    cluster_offset = 0  # Global cluster ID offset

    sub_component_stats = []

    for sub_comp_id in range(num_sub_components):
        sub_mask = (sub_component_labels == sub_comp_id)
        sub_size = sub_mask.sum()

        logger.debug(f"Processing sub-component {sub_comp_id}/{num_sub_components} with {sub_size} buildings")

        # Extract sub-component data
        sub_affinity = affinity_matrix[np.ix_(sub_mask, sub_mask)]
        sub_gat_labels = gat_labels[sub_mask]
        sub_gat_logits = gat_logits[sub_mask] if gat_logits is not None else None

        # Estimate number of clusters for this sub-component
        if n_clusters is None:
            sub_n_clusters = estimate_optimal_clusters(
                sub_affinity,
                oversample_factor=oversample_factor,
                num_buildings=sub_size,
                min_cluster_size=min_cluster_size
            )
        else:
            # Scale n_clusters proportionally to sub-component size
            sub_n_clusters = max(1, int(n_clusters * sub_size / len(embeddings)))

        # Only cluster if sub-component is large enough
        if sub_size >= min_cluster_size and sub_n_clusters > 0:
            # Perform spectral clustering on sub-component
            sub_cluster_assignments = spectral_cluster(
                affinity_matrix=sub_affinity,
                n_clusters=sub_n_clusters,
                random_state=random_state
            )

            # Assign labels to clusters using majority voting
            if use_confidence_weighted_voting and sub_gat_logits is not None:
                _, sub_labels = assign_labels_with_confidence(
                    cluster_assignments=sub_cluster_assignments,
                    gat_labels=sub_gat_labels,
                    gat_logits=sub_gat_logits
                )
            else:
                _, sub_labels = assign_labels_to_clusters(
                    cluster_assignments=sub_cluster_assignments,
                    gat_labels=sub_gat_labels
                )

            # Apply cluster offset to make IDs globally unique
            sub_cluster_assignments_global = sub_cluster_assignments + cluster_offset
            cluster_offset += (sub_cluster_assignments.max() + 1)

            # Store results
            all_cluster_assignments[sub_mask] = sub_cluster_assignments_global
            all_final_labels[sub_mask] = sub_labels

            sub_component_stats.append({
                'sub_component_id': sub_comp_id,
                'size': sub_size,
                'num_clusters': len(np.unique(sub_cluster_assignments)),
                'clustered': True
            })
        else:
            # Sub-component too small: use GAT predictions directly
            all_cluster_assignments[sub_mask] = -1  # Mark as unclustered
            all_final_labels[sub_mask] = sub_gat_labels

            sub_component_stats.append({
                'sub_component_id': sub_comp_id,
                'size': sub_size,
                'num_clusters': 0,
                'clustered': False
            })

            logger.debug(f"Sub-component {sub_comp_id} too small ({sub_size} < {min_cluster_size}), using GAT predictions")

    # Log sub-component statistics
    total_clustered = sum(1 for s in sub_component_stats if s['clustered'])
    logger.debug(f"Clustered {total_clustered}/{num_sub_components} sub-components")

    # Step 4: Filter small clusters globally and revert to GAT predictions
    # Build global cluster_to_label mapping
    cluster_to_label = {}
    for i, assignment in enumerate(all_cluster_assignments):
        if assignment >= 0:  # Valid cluster
            if assignment not in cluster_to_label:
                cluster_to_label[assignment] = all_final_labels[i]

    filtered_cluster_to_label, final_labels, cluster_stats, cleaned_cluster_assignments = filter_small_clusters(
        cluster_assignments=all_cluster_assignments,
        cluster_to_label=cluster_to_label,
        gat_labels=gat_labels,
        min_cluster_size=min_cluster_size
    )

    logger.info(
        "Spectral clustering completed: %d sub-components, %d total clusters, %d valid (size >= %d), %d buildings reverted",
        num_sub_components,
        cluster_stats['total_clusters'],
        cluster_stats['valid_clusters'],
        min_cluster_size,
        cluster_stats['buildings_reverted']
    )

    # Return cleaned cluster assignments (with negative IDs for filtered clusters)
    # This ensures visualization only shows valid clusters
    return cleaned_cluster_assignments, final_labels, filtered_cluster_to_label, affinity_matrix, cluster_stats
