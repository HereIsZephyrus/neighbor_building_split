# Building Pattern Segmentation

A Python-based tool for identifying and segmenting building patterns within districts using deep learning semantic segmentation or classical image segmentation methods. This system processes district and building shapefiles to recognize similar building patterns and group them into regions.

## Features

- **CNN-Based Semantic Segmentation**: DeepLabV3 with ResNet34 backbone for end-to-end building pattern segmentation
- **Multi-scale Feature Extraction**: Combines pre-trained CNN features (ResNet18/VGG16) with handcrafted spatial, height, and geometric features
- **Classical Segmentation**: Uses SLIC superpixels and clustering (K-means/DBSCAN) for pattern recognition
- **Instance Segmentation**: CNN feature-based clustering to separate individual building groups
- **Extensible Architecture**: Abstract interface allows easy integration of custom deep learning models
- **Geospatial Processing**: Full GDAL/QGIS integration with 1m resolution rasterization
- **Automated Workflow**: Processes multiple districts and merges results with continuous cluster IDs

## Workflow

### CNN-Based Segmentation (Recommended)
1. Load district polygons and building footprints from shapefiles
2. For each district:
   - Extract buildings within the district boundary
   - Rasterize buildings at 1m resolution with Floor heights as pixel values
   - Perform semantic segmentation using DeepLabV3-ResNet34
   - Extract CNN features from intermediate layers
   - Cluster features to separate building instances (DBSCAN/K-means)
   - Vectorize segmentation results back to polygons
3. Merge all district segments into a single shapefile with continuous cluster IDs

### Classical Segmentation (Legacy)
1. Load district polygons and building footprints from shapefiles
2. For each district:
   - Extract buildings within the district boundary
   - Rasterize buildings at 1m resolution with Floor heights as pixel values
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
BUILDING=/path/to/buildings.shp         # Building polygons with 'Floor' attribute

# Output directory
OUTPUT_DIR=./output

# Segmentation parameters
N_SEGMENTS=100                          # Number of SLIC superpixels
COMPACTNESS=10.0                        # SLIC compactness (higher = more regular shapes)
SIMILARITY_THRESHOLD=0.5                # Clustering similarity threshold

# Segmentation method (choose one)
USE_CNN_SEGMENTATION=true               # Use CNN-based semantic segmentation (recommended)

# CNN Segmentation parameters (when USE_CNN_SEGMENTATION=true)
CNN_NUM_CLASSES=2                       # Number of classes (2 for binary: background/building)
CNN_CONFIDENCE_THRESHOLD=0.5            # Confidence threshold for predictions (0.0-1.0)
CNN_CLUSTERING_METHOD=dbscan            # Instance clustering: dbscan or kmeans
CNN_N_CLUSTERS=0                        # Number of clusters for kmeans (0 for auto)

# Classical segmentation parameters (when USE_CNN_SEGMENTATION=false)
USE_CNN_FEATURES=true                   # Enable CNN feature extraction
CNN_MODEL=resnet34                      # CNN model: resnet18 or vgg16
```

### Choosing Segmentation Method

**CNN-Based Segmentation** (`USE_CNN_SEGMENTATION=true`):
- **Pros**: More accurate, end-to-end learning, better boundary detection
- **Cons**: Requires GPU for good performance, higher memory usage
- **Best for**: Complex building patterns, high-quality results

**Classical Segmentation** (`USE_CNN_SEGMENTATION=false`):
- **Pros**: Faster on CPU, lower memory usage, interpretable features
- **Cons**: Less accurate boundaries, requires parameter tuning
- **Best for**: Quick prototyping, limited computational resources
```

### Input Data Requirements

**District Shapefile (`DISTRICT`)**:
- Geometry type: Polygon or MultiPolygon
- CRS: Any (will be reprojected to EPSG:32650)
- Attributes: Any attributes will be copied to output segments

**Building Shapefile (`BUILDING`)**:
- Geometry type: Polygon or MultiPolygon
- CRS: Any (will be reprojected to EPSG:32650)
- **Required attribute**: `Floor` (numeric) - building height in floors

## Usage

### Basic Usage

```bash
# Activate conda environment
conda activate spliter

# Run segmentation
python -m src
```

### Voronoi Boundary Generation

Generate Voronoi-like boundaries using dilation method:

```bash
# Generate Voronoi boundaries for districts
python -m src --generate-voronoi-diagram --district-path /path/to/district.shp
```

This mode:
1. Identifies connected building components (using 8-connectivity)
2. Generates Voronoi partitions through iterative dilation
3. Extracts boundaries between different Voronoi regions
4. Converts boundaries to vector line features

**Output Files** (saved to `output/voronoi_diagrams/`):
- `district_{id}_boundaries.shp`: Boundary line features
- `district_{id}_voronoi.tif`: Voronoi partition raster (for debugging)

**Line Feature Attributes**:
- `district_id`: District identifier
- `length`: Line length in meters
- Original district attributes (inherited)

### Raster Generation for Training

Generate rasters for training data preparation:

```bash
# Generate rasters for manual labeling
python -m src --generate-raster-for-training --district-path /path/to/district.shp
```

Output rasters saved to `output/raw_rasters/`.

### Output

The script generates:
- `district_segments.shp`: Segmented regions with continuous cluster IDs (segmentation mode)
- `district_*_boundaries.shp`: Voronoi boundary lines (Voronoi mode)
- `*.log`: Detailed execution log

**Output Shapefile Attributes** (Segmentation mode):
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
│   ├── vectorizer.py            # Raster to vector conversion
│   └── voronoi_generator.py     # Voronoi diagram generation using dilation
├── segmentation/
│   ├── feature_extractor.py     # CNN + handcrafted feature extraction
│   ├── classical_segmenter.py   # SLIC + clustering implementation
│   ├── cnn_segmenter.py         # DeepLabV3-ResNet34 semantic segmentation
│   └── segmentation_interface.py # Abstract base class for segmenters
├── utils/
│   ├── config.py                # Configuration management
│   └── logger.py                # Logging utilities
└── __main__.py                  # Main orchestration script
```

### Key Components

**Voronoi Generator**:
- **Connected Component Analysis**: Identifies building clusters using 8-connectivity
- **Dilation-based Voronoi**: Iteratively expands building regions to partition space
- **Boundary Extraction**: Detects contact lines between adjacent Voronoi regions
- **Vector Conversion**: Converts raster boundaries to line features with simplification
- **District-constrained**: Respects district boundaries during partitioning

**CNN Segmenter**:
- **DeepLabV3 Architecture**: State-of-the-art semantic segmentation
- **ResNet34 Backbone**: Pre-trained on ImageNet for transfer learning
- **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale context
- **Instance Clustering**: DBSCAN/K-means on CNN features for instance separation
- **GPU Accelerated**: Automatic CUDA detection and usage

**Feature Extractor**:
- **CNN Features**: Pre-trained ResNet18/VGG16 for spatial pattern recognition
- **Density Features**: Multi-scale building coverage ratios
- **Height Features**: Mean and standard deviation of Floor values at multiple scales
- **Geometric Features**: Edge detection, corner detection, gradient magnitude

**Classical Segmenter**:
- **SLIC Superpixels**: Over-segments image into homogeneous regions
- **Feature Clustering**: Groups superpixels using K-means or DBSCAN
- **Extensible Interface**: Implements `BaseSegmenter` for easy algorithm swapping

## Training Your Own Model

The CNN segmenter supports supervised training to learn custom building patterns. See [Training Guide](docs/TRAINING_GUIDE.md) for detailed instructions.

### Quick Start

```bash
# 1. Prepare training data from existing segmentation output
python scripts/prepare_training_data.py copy \
    --raster-dir output/debug_rasters \
    --output-dir training_data

# 2. Manually label parcels in QGIS/ArcGIS
# Edit training_data/*_label.tif files to assign cluster IDs

# 3. Train the model
python scripts/train_cnn.py \
    --train-data-dir training_data \
    --save-dir models \
    --num-epochs 50

# 4. Use trained model for inference
# Add to .env: TRAINED_MODEL_PATH=./models/best_model.pth
python -m src
```

### Training Data Format

Training requires paired raster and label files:
- **Input**: `district_X_raster.tif` (building heights)
- **Label**: `district_X_label.tif` (cluster IDs: 1, 2, 3, ...)

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for complete workflow.

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

### CNN Segmentation Parameters

- **CNN_NUM_CLASSES**: Number of segmentation classes
  - `2`: Binary (background/building) - recommended for most cases
  - `>2`: Multi-class for different building types (requires labeled training data)

- **CNN_CONFIDENCE_THRESHOLD**: Minimum confidence for predictions (0.3-0.7 recommended)
  - Lower: Include more uncertain predictions
  - Higher: Only high-confidence predictions

- **CNN_CLUSTERING_METHOD**: Instance separation method
  - `dbscan`: Density-based, automatically determines number of clusters
  - `kmeans`: Fixed number of clusters (set with `CNN_N_CLUSTERS`)

- **SIMILARITY_THRESHOLD**: For DBSCAN clustering (0.3-0.7 recommended)
  - Lower: More clusters, stricter similarity
  - Higher: Fewer clusters, looser similarity

### Classical Segmentation Parameters

- **N_SEGMENTS**: Higher values create more fine-grained segments (50-200 recommended)
- **COMPACTNESS**: Controls SLIC superpixel regularity (5-20 recommended)
  - Lower: Follows image boundaries more closely
  - Higher: More regular, compact superpixels

### Performance

- Resolution is fixed at 1m for balance between detail and performance
- CNN segmentation benefits greatly from GPU acceleration
- Classical segmentation can run efficiently on CPU
- Processing time scales with district area and building density

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce feature resolution or process fewer districts at once
# Consider reducing N_SEGMENTS or disabling CNN features
```

**2. Missing 'Floor' Attribute**
```bash
# Ensure building shapefile has 'Floor' field with numeric values
# Add default Floor values if missing:
# buildings_gdf['Floor'] = 1
```

**3. CRS Mismatch**
```bash
# Script automatically reprojects to EPSG:32650
# Verify input CRS is correctly defined in shapefiles
```

**4. Empty Segmentation Results**
```bash
# Check if buildings intersect districts
# Verify building 'Floor' values are > 0
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
