"""Configuration loader for environment variables."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration class to load and validate environment variables."""

    def __init__(
        self,
        env_path: str = ".env",
        district_path: Optional[str] = None,
        generate_raw_raster: bool = False,
        generate_voronoi_diagram: bool = False,
        visualize_voronoi: bool = False,
        viz_interval: int = 1,
        debug_voronoi: bool = False,
    ):
        """
        Initialize configuration from environment file and optional overrides.

        Args:
            env_path: Path to .env file (default: ".env")
            district_path: Override district path (takes precedence over .env)
            generate_raw_raster: Whether to generate raw raster outputs
            generate_voronoi_diagram: Whether to generate Voronoi diagrams
            visualize_voronoi: Whether to visualize Voronoi dilation process
            viz_interval: Show visualization every N iterations
            debug_voronoi: If True, step through iterations with SPACE key
        """
        load_dotenv(env_path)

        # Store parameter overrides
        self.district_path = Path(district_path)
        self.generate_raw_raster = generate_raw_raster
        self.generate_voronoi_diagram = generate_voronoi_diagram
        self.visualize_voronoi = visualize_voronoi
        self.viz_interval = viz_interval
        self.debug_voronoi = debug_voronoi
        self.image_dir = self.output_dir / "raw_rasters"
        if self.generate_voronoi_diagram:
            self.voronoi_dir = self.output_dir / "voronoi_diagrams"
        else:
            self.voronoi_dir = None

    @property
    def building_path(self) -> Path:
        """Get building shapefile path (parameter override takes precedence)."""
        return Path(os.getenv("BUILDING", ""))

    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        output = os.getenv("OUTPUT_DIR", "./output")
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Training parameters
    @property
    def train_data_dir(self) -> Path:
        """Get training data directory path."""
        train_dir = os.getenv("TRAIN_DATA_DIR", "./training_data")
        return Path(train_dir)

    @property
    def model_save_path(self) -> Path:
        """Get model save directory path."""
        model_dir = os.getenv("MODEL_SAVE_PATH", "./models")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "district_path": str(self.district_path),
            "output_dir": str(self.output_dir),
            "train_data_dir": str(self.train_data_dir),
            "model_save_path": str(self.model_save_path),
            "building_path": str(self.building_path)
        }
