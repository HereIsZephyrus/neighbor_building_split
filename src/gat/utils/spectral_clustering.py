"""Spectral clustering utilities for building clustering.

Combines GAT embeddings, building features, and adjacency matrix
to perform spectral clustering and assign GAT labels to clusters.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
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
    min_clusters: int = 2
) -> int:
    """
    Estimate optimal number of clusters using eigenvalue analysis.

    Args:
        affinity_matrix: Affinity matrix (N, N)
        max_clusters: Maximum number of clusters to consider
        min_clusters: Minimum number of clusters

    Returns:
        optimal_k: Estimated optimal number of clusters
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

    optimal_k = np.argmax(eigengaps) + 1
    optimal_k = max(min_clusters, min(optimal_k, max_clusters))

    logger.info("Estimated optimal number of clusters: %d", optimal_k)

    return optimal_k


def perform_spectral_clustering_pipeline(
    embeddings: np.ndarray,
    features: np.ndarray,
    adjacency_matrix: pd.DataFrame,
    gat_labels: np.ndarray,
    n_clusters: Optional[int] = None,
    embedding_weight: float = 0.5,
    feature_weight: float = 0.3,
    distance_weight: float = 0.2,
    distance_scale: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Complete spectral clustering pipeline.

    Args:
        embeddings: GAT embeddings (N, D_emb)
        features: Original features (N, D_feat)
        adjacency_matrix: Distance-based adjacency (N, N)
        gat_labels: GAT predicted labels (N,)
        n_clusters: Number of clusters (auto-estimate if None)
        embedding_weight: Weight for embedding similarity
        feature_weight: Weight for feature similarity
        distance_weight: Weight for distance-based adjacency
        distance_scale: Scale factor for distance conversion
        random_state: Random seed

    Returns:
        cluster_assignments: Spectral cluster assignments (N,)
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

    # Step 2: Estimate optimal clusters if not provided
    if n_clusters is None:
        n_clusters = estimate_optimal_clusters(affinity_matrix)

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

    logger.info("Spectral clustering pipeline completed successfully")

    return cluster_assignments, final_labels, cluster_to_label, affinity_matrix

