# Building Pattern Segmentation & LCZ Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

!Notice This Doc archived by AI

[中文文档](README_zh.md) | English

A two-stage urban building classification system using **Graph Attention Networks (GAT)** and **Spectral Clustering** for Local Climate Zone (LCZ) classification. This project processes building footprints to generate spatial adjacency relationships via Voronoi diagrams, then applies deep learning for accurate, spatially-consistent building classification.

## Overview

Urban building classification is crucial for climate studies, urban planning, and energy modeling. This project addresses the challenge of classifying buildings into Local Climate Zones (LCZ1-9) while maintaining spatial consistency—nearby similar buildings should belong to the same category.

### Two-Stage Approach

```
Stage 1: GAT Classification          Stage 2: Spectral Clustering
┌─────────────────────────┐         ┌─────────────────────────────────┐
│ Discriminative Features │         │ Morphological Features          │
│ (height, albedo, etc.)  │ ──────► │ (area, shape, orientation)      │
│                         │         │                                 │
│ "What type of building?"│         │ "Which buildings belong together?"│
└─────────────────────────┘         └─────────────────────────────────┘
           │                                       │
           ▼                                       ▼
    Initial Predictions              Spatially-Consistent Final Labels
```

**Key Innovation**: Task decoupling—GAT focuses on classification accuracy while spectral clustering ensures spatial coherence through confidence-weighted majority voting.

## Features

- **Voronoi-Based Adjacency**: Generate building adjacency relationships using morphological dilation
- **Distance-Aware GAT**: Graph Attention Network with edge features encoding spatial distances
- **Two-Stage Classification**: Combines discriminative classification with spatial smoothing
- **Confidence-Weighted Voting**: High-confidence GAT predictions have more influence on cluster labels
- **MPI Parallel Processing**: Distributed processing for large-scale datasets
- **LCZ Similarity Loss**: Custom loss function aware of semantic relationships between LCZ classes
- **Connected Component Handling**: Automatic separation and processing of disconnected building groups

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- MPI (optional, for parallel processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/neighbor_building_split.git
cd neighbor_building_split

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
pip install torch-geometric torch-scatter torch-sparse
```

### Environment Configuration

Create a `.env` file in the project root:

```env
DISTRICT=/path/to/districts.shp
BUILDINGS=/path/to/buildings.shp
OUTPUT_DIR=/path/to/output
```

## Quick Start

### 1. Generate Voronoi Diagrams & Adjacency Matrices

```bash
# Single-threaded processing
python -m src.extractor \
    --generate-voronoi-diagram \
    --district-path /path/to/districts.shp

# MPI parallel processing (8 processes)
mpirun -n 8 python -m src.extractor \
    --generate-voronoi-diagram \
    --district-path /path/to/districts.shp \
    --use-mpi
```

### 2. Train GAT Model

```bash
# Cross-validation training
python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings /path/to/buildings.shp \
    --sample-districts /path/to/districts.shp \
    --output-root-dir output/gat \
    --config src/gat/training_config.yaml \
    --mode cv

# Final model training (on all data)
python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings /path/to/buildings.shp \
    --sample-districts /path/to/districts.shp \
    --output-root-dir output/gat \
    --mode final
```

### 3. Run Inference

```bash
python -m src.gat --inference \
    --model-path output/gat/models/final_model.pth \
    --adjacency-dir output/voronoi \
    --building-path /path/to/buildings.shp \
    --output-root-dir output/predictions
```

## Module Overview

### Extractor Module (`src/extractor/`)

Generates Voronoi diagrams from building footprints and computes spatial adjacency matrices.

**Workflow:**
1. Load district and building shapefiles
2. Rasterize building footprints within each district
3. Generate Voronoi partition via morphological dilation
4. Vectorize Voronoi polygons
5. Compute adjacency matrix (distances between adjacent buildings)

**Key Components:**
- `VoronoiGenerator`: Morphological dilation-based Voronoi generation
- `Rasterizer`: Building footprint rasterization
- `ShapefileReader`: Spatial data loading and filtering

**Output:**
- `district_{id}_voronoi.shp`: Voronoi polygon shapefile
- `district_{id}_adjacency.pkl`: Building adjacency matrix (distances)

### GAT Module (`src/gat/`)

Graph Attention Network for building classification with spectral clustering post-processing.

**Training Workflow:**
1. Load building features and adjacency matrices
2. Construct graph with nodes (buildings) and edges (adjacency)
3. Train GAT model with similarity-aware loss
4. K-fold cross-validation for hyperparameter tuning

**Inference Workflow:**
1. Load trained model and building data
2. GAT forward pass → embeddings + initial predictions
3. Extract clustering features (morphological)
4. Perform spectral clustering on each connected component
5. Confidence-weighted majority voting within clusters
6. Output final spatially-consistent predictions

**Key Components:**
- `EdgeConvLayer`: Distance-aware graph attention layer
- `GAT`: Multi-layer graph attention network
- `spectral_clustering.py`: Two-stage clustering pipeline
- `SimilarityAwareCrossEntropyLoss`: LCZ-aware loss function

## Configuration

### Training Configuration (`src/gat/training_config.yaml`)

```yaml
model:
  hidden_dim: 32
  num_layers: 3
  num_heads: 8
  dropout: 0.6

training:
  epochs: 2000
  lr: 0.005
  patience: 120
  k_fold: 5
  lambda_smooth: 0.3

spectral_clustering:
  embedding_weight: 0.6
  feature_weight: 0.2
  distance_weight: 0.2
  min_cluster_size: 5
  use_confidence_weighted_voting: true

similarity_loss:
  enabled: true
  temperature: 0.05
```

### Feature Configuration (`src/gat/features_config.yaml`)

```yaml
# Features for GAT classification (discriminative)
gat_features:
  - height
  - albedo
  - hwratio
  - area

# Features for spectral clustering (morphological)
clustering_features:
  - height
  - area
  - perimeter
  - orientatio
  - elongation
  - concavity
  - circularit
```

## Project Structure

```
neighbor_building_split/
├── src/
│   ├── extractor/              # Voronoi & adjacency module
│   │   ├── __main__.py         # CLI entry point
│   │   ├── processor.py        # District processing orchestrator
│   │   ├── converter/
│   │   │   ├── voronoi_generator.py
│   │   │   └── rasterizer.py
│   │   ├── reader/
│   │   │   └── shapefile_reader.py
│   │   └── utils/
│   │       ├── adjacency.py
│   │       └── config.py
│   │
│   └── gat/                    # GAT classification module
│       ├── __main__.py         # CLI entry point
│       ├── train.py            # Training orchestrator
│       ├── inference.py        # Inference pipeline
│       ├── models/
│       │   ├── gat.py          # GAT model
│       │   └── gat_layer.py    # EdgeConv attention layer
│       ├── training/
│       │   ├── trainer.py
│       │   ├── similarity_loss.py
│       │   └── config.py
│       ├── utils/
│       │   ├── spectral_clustering.py
│       │   └── feature_extractor.py
│       ├── training_config.yaml
│       └── features_config.yaml
│
├── docs/                       # Documentation (Chinese)
├── scripts/                    # Utility scripts
├── test/                       # Test files
├── requirements.txt
└── README.md
```

## Data Requirements

### Building Shapefile

Required attributes:
| Field | Description | Used By |
|-------|-------------|---------|
| `id` | Unique building identifier | Both |
| `height` | Building height (meters) | GAT, Clustering |
| `albedo` | Surface albedo | GAT |
| `hwratio` | Height-to-width ratio | GAT |
| `area` | Footprint area (m²) | GAT, Clustering |
| `perimeter` | Perimeter (meters) | Clustering |
| `orientatio` | Orientation (degrees) | Clustering |
| `lcz` | Ground truth label (training only) | Training |

### District Shapefile

Required attributes:
| Field | Description |
|-------|-------------|
| `FID` or `fid` | District identifier |
| `geometry` | District polygon |

## Advanced Usage

### MPI Parallel Training

```bash
mpirun -n 8 python -m src.gat --train \
    --adjacency-dir output/voronoi \
    --sample-buildings buildings.shp \
    --sample-districts districts.shp \
    --output-root-dir output/gat \
    --mode cv
```

### Resume Training

```bash
python -m src.gat --train \
    --resume output/gat/checkpoints/checkpoint_epoch_100.pth \
    ...
```

### Custom Clustering Scaler

```bash
python -m src.gat --inference \
    --model-path model.pth \
    --clustering-scaler-path custom_scaler.pkl \
    ...
```

## Outputs

### Training Outputs

```
output/gat/
├── models/
│   ├── best_model.pth          # Best validation model
│   └── final_model.pth         # Final trained model
├── logs/
│   └── training.log
└── runs/                       # TensorBoard logs
```

### Inference Outputs

```
output/predictions/
├── district_{id}_embeddings.pkl        # GAT embeddings
├── district_{id}_building_predictions.gpkg   # Building predictions
├── district_{id}_voronoi_predictions.gpkg    # Voronoi predictions
└── embeddings_summary.pkl              # Summary statistics
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in training config
   - Set `node_threshold` to use mini-batch sampling for large graphs

2. **No Buildings Found**
   - Check CRS consistency between building and district shapefiles
   - Verify spatial intersection between datasets

3. **Poor Classification Accuracy**
   - Adjust `similarity_loss.temperature` (lower = less smoothing)
   - Tune spectral clustering weights (`embedding_weight`, `feature_weight`)
   - Increase training `epochs` or adjust `patience`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

For detailed documentation in Chinese, see:
- [算法思想](docs/算法思想.md) - Algorithm concepts and theory
- [设计细节](docs/设计细节.md) - Architecture and design details
- [实现方法](docs/实现方法.md) - Implementation guide

