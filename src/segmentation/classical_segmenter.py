"""Classical segmentation using SLIC and clustering."""

from typing import Optional
import numpy as np
from skimage import segmentation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from .segmentation_interface import BaseSegmenter
from ..utils.logger import get_logger

logger = get_logger()


class ClassicalSegmenter(BaseSegmenter):
    """Classical segmentation using SLIC superpixels and clustering."""

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 10.0,
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize classical segmenter.

        Args:
            n_segments: Number of SLIC superpixels
            compactness: SLIC compactness parameter
            clustering_method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for k-means (auto if None)
            similarity_threshold: Similarity threshold for merging
        """
        super().__init__()
        self.params = {
            "n_segments": n_segments,
            "compactness": compactness,
            "clustering_method": clustering_method,
            "n_clusters": n_clusters,
            "similarity_threshold": similarity_threshold,
        }
        self.scaler = StandardScaler()
        self.superpixel_labels = None
        self.superpixel_features = None

    def fit(self, features: np.ndarray) -> "ClassicalSegmenter":
        """
        Fit the feature scaler.

        Args:
            features: Feature array (H, W, C)

        Returns:
            Self for method chaining
        """
        # Reshape features to 2D for scaling
        c = features.shape[2]
        features_2d = features.reshape(-1, c)

        # Remove invalid features
        valid_mask = np.all(np.isfinite(features_2d), axis=1)
        if valid_mask.sum() > 0:
            self.scaler.fit(features_2d[valid_mask])
            logger.debug("Fitted feature scaler")
        else:
            logger.warning("No valid features for scaling")

        return self

    def _compute_superpixels(
        self, raster: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute SLIC superpixels.

        Args:
            raster: Input raster array
            mask: Optional mask to constrain superpixels

        Returns:
            Superpixel label array
        """
        logger.debug(
            "Computing SLIC superpixels: n_segments=%d, compactness=%s",
            self.params['n_segments'], self.params['compactness']
        )

        # Apply mask if provided
        if mask is not None:
            raster_masked = raster.copy()
            raster_masked[mask == 0] = 0
        else:
            raster_masked = raster

        # Compute superpixels
        labels = segmentation.slic(
            raster_masked,
            n_segments=self.params["n_segments"],
            compactness=self.params["compactness"],
            start_label=0,
            channel_axis=None,
        )

        # Filter out superpixels outside mask
        if mask is not None:
            labels[mask == 0] = -1

        n_superpixels = len(np.unique(labels[labels >= 0]))
        logger.debug("Generated %d superpixels", n_superpixels)

        return labels

    def _extract_superpixel_features(
        self, features: np.ndarray, superpixel_labels: np.ndarray
    ) -> np.ndarray:
        """
        Extract mean features for each superpixel.

        Args:
            features: Feature array (H, W, C)
            superpixel_labels: Superpixel label array (H, W)

        Returns:
            Superpixel feature array (N_superpixels, C)
        """
        unique_labels = np.unique(superpixel_labels)
        unique_labels = unique_labels[unique_labels >= 0]

        c = features.shape[2]
        features_2d = features.reshape(-1, c)
        labels_1d = superpixel_labels.reshape(-1)

        superpixel_features = np.zeros((len(unique_labels), c))

        for i, label in enumerate(unique_labels):
            mask = labels_1d == label
            if mask.sum() > 0:
                superpixel_features[i] = np.mean(features_2d[mask], axis=0)

        logger.debug("Extracted features for %d superpixels", len(unique_labels))
        return superpixel_features

    def _cluster_superpixels(self, superpixel_features: np.ndarray) -> np.ndarray:
        """
        Cluster superpixels based on features.

        Args:
            superpixel_features: Feature array for superpixels (N, C)

        Returns:
            Cluster labels for each superpixel
        """
        # Scale features
        features_scaled = self.scaler.transform(superpixel_features)

        # Remove invalid features
        valid_mask = np.all(np.isfinite(features_scaled), axis=1)
        if valid_mask.sum() == 0:
            logger.warning("No valid features for clustering")
            return np.zeros(len(superpixel_features), dtype=np.int32)

        # Clustering
        method = self.params["clustering_method"]
        logger.debug("Clustering superpixels using %s", method)

        cluster_labels = np.zeros(len(superpixel_features), dtype=np.int32)

        if method == "kmeans":
            n_clusters = self.params["n_clusters"]
            if n_clusters is None:
                # Auto-determine number of clusters
                n_clusters = min(10, max(2, valid_mask.sum() // 10))
            n_clusters = min(n_clusters, valid_mask.sum())

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels[valid_mask] = kmeans.fit_predict(
                features_scaled[valid_mask]
            )

            logger.debug("K-means clustering with %d clusters", n_clusters)

        elif method == "dbscan":
            eps = self.params["similarity_threshold"]
            dbscan = DBSCAN(eps=eps, min_samples=2)
            cluster_labels[valid_mask] = dbscan.fit_predict(
                features_scaled[valid_mask]
            )

            # Map noise (-1) to separate cluster
            cluster_labels[cluster_labels == -1] = cluster_labels.max() + 1

            logger.debug(
                "DBSCAN clustering found %d clusters",
                len(np.unique(cluster_labels))
            )

        return cluster_labels

    def predict(self, raster_data: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on raster data.

        Args:
            raster_data: Input raster array (H, W)
            features: Feature array for segmentation (H, W, C)

        Returns:
            Labeled array where each region has unique integer ID
        """
        # Create mask for areas with buildings
        mask = (raster_data > 0).astype(np.uint8)

        # Compute superpixels
        superpixel_labels = self._compute_superpixels(raster_data, mask)
        self.superpixel_labels = superpixel_labels

        # Extract superpixel features
        superpixel_features = self._extract_superpixel_features(
            features, superpixel_labels
        )
        self.superpixel_features = superpixel_features

        # Cluster superpixels
        cluster_labels = self._cluster_superpixels(superpixel_features)

        # Map superpixel labels to cluster labels
        result = np.zeros_like(superpixel_labels, dtype=np.int32)
        unique_labels = np.unique(superpixel_labels)
        unique_labels = unique_labels[unique_labels >= 0]

        for i, label in enumerate(unique_labels):
            result[superpixel_labels == label] = cluster_labels[i] + 1

        # Set background to 0
        result[mask == 0] = 0

        n_clusters = len(np.unique(result)) - 1  # Exclude background
        logger.info("Segmentation complete: %d clusters identified", n_clusters)

        return result

