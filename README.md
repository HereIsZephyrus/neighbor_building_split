# Building Pattern Segmentation

A Python-based tool for identifying and segmenting building patterns within districts using classical image segmentation combined with pre-trained CNN features. This system processes district and building shapefiles to recognize similar building patterns and group them into regions.

## Features

- **Multi-scale Feature Extraction**: Combines pre-trained CNN features (ResNet18/VGG16) with handcrafted spatial, height, and geometric features
- **Classical Segmentation**: Uses SLIC superpixels and clustering (K-means/DBSCAN) for pattern recognition
- **Extensible Architecture**: Abstract interface allows easy integration of custom deep learning models
- **Geospatial Processing**: Full GDAL/QGIS integration with 1m resolution rasterization
- **Automated Workflow**: Processes multiple districts and merges results with continuous cluster IDs

## Workflow

1. Load district polygons and building footprints from shapefiles
2. For each district:
   - Extract buildings within the district boundary
   - Rasterize buildings at 1m resolution with floor heights as pixel values
   - Extract multi-scale features (CNN + spatial density + height statistics + geometric patterns)
   - Perform SLIC superpixel segmentation
   - Cluster superpixels based on feature similarity
   - Vectorize segmentation results back to polygons
3. Merge all district segments into a single shapefile with continuous cluster IDs

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate spliter
```

### 2. Install Additional Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -m src --help
```

## Configuration

Create a `.env` file in the project root (use `.env.example` as template):

```bash
# Required paths
DISTRICT=/path/to/district.shp          # District polygon shapefile
BUILDING=/path/to/buildings.shp         # Building polygons with 'floor' attribute

# Output directory
OUTPUT_DIR=./output

# Segmentation parameters
N_SEGMENTS=100                          # Number of SLIC superpixels
COMPACTNESS=10.0                        # SLIC compactness (higher = more regular shapes)
SIMILARITY_THRESHOLD=0.5                # Clustering similarity threshold

# Feature extraction
USE_CNN_FEATURES=true                   # Enable CNN feature extraction
CNN_MODEL=resnet18                      # CNN model: resnet18 or vgg16
```

### Input Data Requirements

**District Shapefile (`DISTRICT`)**:
- Geometry type: Polygon or MultiPolygon
- CRS: Any (will be reprojected to EPSG:32650)
- Attributes: Any attributes will be copied to output segments

**Building Shapefile (`BUILDING`)**:
- Geometry type: Polygon or MultiPolygon
- CRS: Any (will be reprojected to EPSG:32650)
- **Required attribute**: `floor` (numeric) - building height in floors

## Usage

### Basic Usage

```bash
# Activate conda environment
conda activate spliter

# Run segmentation
python -m src
```

### Output

The script generates:
- `district_segments.shp`: Segmented regions with continuous cluster IDs
- `segmentation.log`: Detailed execution log

**Output Shapefile Attributes**:
- `cluster_id`: Unique segment identifier (continuous across all districts)
- `area`: Segment area in square meters
- `building_count`: Number of buildings within the segment
- Original district attributes (copied from input)

## Architecture

### Module Structure

```
src/
├── reader/
│   └── shapefile_reader.py      # Load and filter shapefiles
├── converter/
│   ├── rasterizer.py            # Vector to raster conversion (1m resolution)
│   └── vectorizer.py            # Raster to vector conversion
├── segmentation/
│   ├── feature_extractor.py     # CNN + handcrafted feature extraction
│   ├── classical_segmenter.py   # SLIC + clustering implementation
│   └── segmentation_interface.py # Abstract base class for segmenters
├── utils/
│   ├── config.py                # Configuration management
│   └── logger.py                # Logging utilities
└── __main__.py                  # Main orchestration script
```

### Key Components

**Feature Extractor**:
- **CNN Features**: Pre-trained ResNet18/VGG16 for spatial pattern recognition
- **Density Features**: Multi-scale building coverage ratios
- **Height Features**: Mean and standard deviation of floor values at multiple scales
- **Geometric Features**: Edge detection, corner detection, gradient magnitude

**Classical Segmenter**:
- **SLIC Superpixels**: Over-segments image into homogeneous regions
- **Feature Clustering**: Groups superpixels using K-means or DBSCAN
- **Extensible Interface**: Implements `BaseSegmenter` for easy algorithm swapping

## Future Expansion

### Adding Deep Learning Models

The architecture supports custom segmentation models through the `BaseSegmenter` interface:

```python
from src.segmentation.segmentation_interface import BaseSegmenter
import numpy as np

class DeepLearningSegmenter(BaseSegmenter):
    """Custom deep learning segmenter."""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model = load_your_model(model_path)
    
    def fit(self, features: np.ndarray) -> "DeepLearningSegmenter":
        # Optional: Fine-tune model
        return self
    
    def predict(self, raster_data: np.ndarray, features: np.ndarray) -> np.ndarray:
        # Run model inference
        return self.model.predict(raster_data)
```

Then use it in the main script:

```python
from your_module import DeepLearningSegmenter

# Replace classical segmenter
segmenter = DeepLearningSegmenter(model_path="path/to/model.pth")
```

## Parameters Tuning

### Segmentation Quality

- **N_SEGMENTS**: Higher values create more fine-grained segments (50-200 recommended)
- **COMPACTNESS**: Controls SLIC superpixel regularity (5-20 recommended)
  - Lower: Follows image boundaries more closely
  - Higher: More regular, compact superpixels

### Clustering

- **SIMILARITY_THRESHOLD**: For DBSCAN clustering (0.3-0.7 recommended)
  - Lower: More clusters, stricter similarity
  - Higher: Fewer clusters, looser similarity

### Performance

- Resolution is fixed at 1m for balance between detail and performance
- CNN features can be disabled (`USE_CNN_FEATURES=false`) for faster processing
- Processing time scales with district area and building density

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce feature resolution or process fewer districts at once
# Consider reducing N_SEGMENTS or disabling CNN features
```

**2. Missing 'floor' Attribute**
```bash
# Ensure building shapefile has 'floor' field with numeric values
# Add default floor values if missing:
# buildings_gdf['floor'] = 1
```

**3. CRS Mismatch**
```bash
# Script automatically reprojects to EPSG:32650
# Verify input CRS is correctly defined in shapefiles
```

**4. Empty Segmentation Results**
```bash
# Check if buildings intersect districts
# Verify building 'floor' values are > 0
# Check log file for detailed error messages
```

## Development

### Code Quality

The codebase follows strict quality standards:
- **Pylint score**: Target 9.0+
- **Type hints**: All function signatures
- **Docstrings**: Google style for all public methods
- **Comments**: English only

### Running Linter

```bash
conda activate spliter
pylint src/
```

### Testing

```bash
# Run with small test dataset first
# Check output and log files for issues
python -m src
```

## Requirements

- Python 3.12+
- GDAL 3.11+
- CUDA-capable GPU (optional, for faster CNN feature extraction)
- At least 8GB RAM recommended

## License

See LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{building_pattern_segmentation,
  title={Building Pattern Segmentation Tool},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Support

For issues and questions:
1. Check the log file in `OUTPUT_DIR/segmentation.log`
2. Verify input data format and configuration
3. Open an issue on GitHub with log file and error details
