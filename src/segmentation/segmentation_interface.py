"""Abstract interface for segmentation algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseSegmenter(ABC):
    """Abstract base class for segmentation algorithms."""

    def __init__(self):
        """Initialize base segmenter."""
        self.params = {}

    @abstractmethod
    def fit(self, features: np.ndarray) -> "BaseSegmenter":
        """
        Train or configure the segmenter with features.

        Args:
            features: Feature array (can be used for training)

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def predict(self, raster_data: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on raster data.

        Args:
            raster_data: Input raster array
            features: Feature array for segmentation

        Returns:
            Labeled array where each region has unique integer ID
        """

    def get_params(self) -> Dict[str, Any]:
        """
        Get segmenter parameters.

        Returns:
            Dictionary of parameters
        """
        return self.params.copy()

    def set_params(self, **params) -> "BaseSegmenter":
        """
        Set segmenter parameters.

        Args:
            **params: Parameter key-value pairs

        Returns:
            Self for method chaining
        """
        self.params.update(params)
        return self

    def fit_predict(self, raster_data: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one call.

        Args:
            raster_data: Input raster array
            features: Feature array for segmentation

        Returns:
            Labeled array where each region has unique integer ID
        """
        self.fit(features)
        return self.predict(raster_data, features)

