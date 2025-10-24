"""Configuration loader for environment variables."""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration class to load and validate environment variables."""

    def __init__(self, env_path: str = ".env"):
        """
        Initialize configuration from environment file.

        Args:
            env_path: Path to .env file (default: ".env")
        """
        load_dotenv(env_path)
        self._validate_required_vars()

    def _validate_required_vars(self) -> None:
        """Validate that required environment variables are set."""
        required_vars = ["DISTRICT", "BUILDING"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    @property
    def district_path(self) -> Path:
        """Get district shapefile path."""
        return Path(os.getenv("DISTRICT", ""))

    @property
    def building_path(self) -> Path:
        """Get building shapefile path."""
        return Path(os.getenv("BUILDING", ""))

    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        output = os.getenv("OUTPUT_DIR", "./output")
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def n_segments(self) -> int:
        """Get number of SLIC segments."""
        return int(os.getenv("N_SEGMENTS", "100"))

    @property
    def compactness(self) -> float:
        """Get SLIC compactness parameter."""
        return float(os.getenv("COMPACTNESS", "10.0"))

    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold for clustering."""
        return float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    @property
    def use_cnn_features(self) -> bool:
        """Whether to use CNN features."""
        return os.getenv("USE_CNN_FEATURES", "true").lower() == "true"

    @property
    def cnn_model(self) -> str:
        """Get CNN model name."""
        return os.getenv("CNN_MODEL", "resnet18")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "district_path": str(self.district_path),
            "building_path": str(self.building_path),
            "output_dir": str(self.output_dir),
            "n_segments": self.n_segments,
            "compactness": self.compactness,
            "similarity_threshold": self.similarity_threshold,
            "use_cnn_features": self.use_cnn_features,
            "cnn_model": self.cnn_model,
        }
