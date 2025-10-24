"""Feature extraction using CNN and handcrafted features."""

from typing import Optional, Tuple
import numpy as np
import torch
import torchvision.models as models
from scipy import ndimage
from skimage import feature, filters
from PIL import Image
from ..utils.logger import get_logger

logger = get_logger()


class FeatureExtractor:
    """Extract features from rasterized building data."""

    def __init__(
        self,
        use_cnn: bool = True,
        cnn_model_name: str = "resnet18",
        device: Optional[str] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            use_cnn: Whether to use CNN features
            cnn_model_name: Name of pre-trained CNN model
            device: Device for torch (cuda/cpu), auto-detect if None
        """
        self.use_cnn = use_cnn
        self.cnn_model_name = cnn_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if self.use_cnn:
            self._load_cnn_model()

    def _load_cnn_model(self) -> None:
        """Load pre-trained CNN model."""
        logger.info("Loading pre-trained %s model", self.cnn_model_name)

        if self.cnn_model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        elif self.cnn_model_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT
            self.model = models.vgg16(weights=weights)
        else:
            raise ValueError(f"Unsupported CNN model: {self.cnn_model_name}")

        # Remove final classification layer
        if self.cnn_model_name == "resnet18":
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        elif self.cnn_model_name == "vgg16":
            self.model = self.model.features

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("CNN model loaded on %s", self.device)

    def extract_cnn_features(
        self, raster: np.ndarray, downsample_factor: int = 4
    ) -> np.ndarray:
        """
        Extract CNN features from raster data.

        Args:
            raster: Input raster array (H, W)
            downsample_factor: Factor to downsample features

        Returns:
            Feature map array (H', W', C)
        """
        if not self.use_cnn or self.model is None:
            return np.array([])

        # Normalize raster to 0-1 range
        raster_norm = raster.copy()
        if raster_norm.max() > 0:
            raster_norm = raster_norm / raster_norm.max()

        # Convert to 3-channel image
        raster_3ch = np.stack([raster_norm] * 3, axis=-1)
        img = Image.fromarray((raster_3ch * 255).astype(np.uint8))

        # Resize to divisible by 32 for CNN
        h, w = raster.shape
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        # Convert back to numpy
        features = features.squeeze(0).cpu().numpy()
        features = np.transpose(features, (1, 2, 0))  # C, H, W -> H, W, C

        # Resize back to original dimensions (downsampled)
        target_h = h // downsample_factor
        target_w = w // downsample_factor

        # Interpolate each channel
        features_resized = np.zeros((target_h, target_w, features.shape[2]))
        for i in range(features.shape[2]):
            features_resized[:, :, i] = np.array(
                Image.fromarray(features[:, :, i]).resize(
                    (target_w, target_h), Image.BILINEAR
                )
            )

        logger.debug("Extracted CNN features with shape %s", features_resized.shape)
        return features_resized

    def extract_density_features(
        self, raster: np.ndarray, window_sizes: Tuple[int, ...] = (5, 10, 20)
    ) -> np.ndarray:
        """
        Extract spatial density features at multiple scales.

        Args:
            raster: Input raster array
            window_sizes: Window sizes for density calculation

        Returns:
            Density feature array (H, W, N_scales)
        """
        h, w = raster.shape
        density_features = np.zeros((h, w, len(window_sizes)))

        # Binary mask of buildings
        building_mask = (raster > 0).astype(np.float32)

        for i, window_size in enumerate(window_sizes):
            # Calculate local density using uniform filter
            kernel = np.ones((window_size, window_size))
            density = ndimage.convolve(building_mask, kernel, mode="constant") / (
                window_size * window_size
            )
            density_features[:, :, i] = density

        logger.debug("Extracted density features at %d scales", len(window_sizes))
        return density_features

    def extract_height_features(
        self, raster: np.ndarray, window_sizes: Tuple[int, ...] = (5, 10, 20)
    ) -> np.ndarray:
        """
        Extract height statistics features.

        Args:
            raster: Input raster array with floor values
            window_sizes: Window sizes for statistics calculation

        Returns:
            Height feature array (H, W, N_scales*2) for mean and std
        """
        h, w = raster.shape
        height_features = np.zeros((h, w, len(window_sizes) * 2))

        for i, window_size in enumerate(window_sizes):
            kernel = np.ones((window_size, window_size))

            # Mean height
            mean_height = ndimage.convolve(raster, kernel, mode="constant") / (
                window_size * window_size
            )
            height_features[:, :, i * 2] = mean_height

            # Standard deviation
            raster_sq = raster ** 2
            mean_sq = ndimage.convolve(raster_sq, kernel, mode="constant") / (
                window_size * window_size
            )
            std_height = np.sqrt(np.maximum(mean_sq - mean_height ** 2, 0))
            height_features[:, :, i * 2 + 1] = std_height

        logger.debug("Extracted height features at %d scales", len(window_sizes))
        return height_features

    def extract_geometric_features(self, raster: np.ndarray) -> np.ndarray:
        """
        Extract geometric pattern features using edge and corner detection.

        Args:
            raster: Input raster array

        Returns:
            Geometric feature array (H, W, 3) for edges, corners, and gradient
        """
        h, w = raster.shape
        geom_features = np.zeros((h, w, 3))

        # Edge detection (Sobel)
        edges = filters.sobel(raster)
        geom_features[:, :, 0] = edges

        # Corner detection (Harris)
        corners = feature.corner_harris(raster, sigma=1)
        geom_features[:, :, 1] = corners

        # Gradient magnitude
        grad_y, grad_x = np.gradient(raster)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        geom_features[:, :, 2] = gradient_mag

        logger.debug("Extracted geometric features (edges, corners, gradients)")
        return geom_features

    def extract_features(
        self, raster: np.ndarray, downsample_factor: int = 4
    ) -> np.ndarray:
        """
        Extract all features from raster data.

        Args:
            raster: Input raster array
            downsample_factor: Factor to downsample features

        Returns:
            Combined feature array (H', W', N_features)
        """
        logger.info("Extracting features from raster")

        feature_list = []

        # Handcrafted features
        density = self.extract_density_features(raster)
        height = self.extract_height_features(raster)
        geometric = self.extract_geometric_features(raster)

        # Downsample handcrafted features
        h_target = raster.shape[0] // downsample_factor
        w_target = raster.shape[1] // downsample_factor

        for feat in [density, height, geometric]:
            feat_downsampled = np.zeros((h_target, w_target, feat.shape[2]))
            for i in range(feat.shape[2]):
                feat_downsampled[:, :, i] = np.array(
                    Image.fromarray(feat[:, :, i]).resize(
                        (w_target, h_target), Image.BILINEAR
                    )
                )
            feature_list.append(feat_downsampled)

        # CNN features
        if self.use_cnn:
            cnn_features = self.extract_cnn_features(raster, downsample_factor)
            if cnn_features.size > 0:
                feature_list.append(cnn_features)

        # Concatenate all features
        features = np.concatenate(feature_list, axis=-1)

        logger.info("Extracted total %d features", features.shape[2])
        return features

