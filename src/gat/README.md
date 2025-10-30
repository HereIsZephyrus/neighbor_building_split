# GAT Module for Building Clustering

Graph Attention Network (GAT) implementation for building clustering based on [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT).

## Overview

This module implements a GAT model to classify buildings in urban districts based on their spatial relationships and geometric features. The model:

- Takes building graphs as input (nodes = buildings, edges = adjacency/similarity)
- Extracts 12 geometric and shape features from building geometries
- Learns node representations using multi-head attention mechanism
- Generates embeddings for downstream spectral clustering

## Architecture

- **Model**: 3-layer GAT with multi-head attention
  - Layer 1: 12 → 64 (8 heads, concatenated)
  - Layer 2: 512 → 64 (8 heads, concatenated)
  - Layer 3: 512 → num_classes (1 head, averaged)
- **Activation**: ELU
- **Dropout**: 0.6
- **Optimizer**: Adam (lr=5e-3, weight_decay=5e-4)
- **Embedding dimension**: 512 (for clustering)

## Installation

Install required dependencies:

```bash
pip install torch-geometric torch-scatter torch-sparse tensorboard
```

Or use the project's requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train GAT model on building graphs:

```bash
# Basic training
python -m src.gat \
    --adjacency-dir output/voronoi \
    --building-shapefile path/to/buildings.shp

# Advanced options
python -m src.gat \
    --adjacency-dir output/voronoi \
    --building-shapefile path/to/buildings.shp \
    --hidden-dim 64 \
    --num-layers 3 \
    --num-heads 8 \
    --dropout 0.6 \
    --lr 0.005 \
    --epochs 200 \
    --batch-size 1024 \
    --device cuda
```

### 2. Monitoring with TensorBoard

```bash
tensorboard --logdir runs/gat
```

Then open http://localhost:6006 in your browser.

### 3. Inference (Generate Embeddings)

Generate embeddings for downstream clustering:

```bash
python -m src.gat.inference \
    --checkpoint models/gat/best_model.pth \
    --adjacency-dir output/voronoi \
    --building-shapefile path/to/buildings.shp \
    --output-root-dir output/gat/embeddings
```

This creates:
- `district_{id}_embeddings.pkl` for each district
- `embeddings_summary.pkl` with overall statistics

### 4. Resume Training

```bash
python -m src.gat \
    --resume models/gat/checkpoint_epoch_100.pth \
    --building-shapefile path/to/buildings.shp
```

## Configuration

You can use a JSON config file instead of command-line arguments:

```json
{
  "data_dir": "output/voronoi",
  "building_shapefile": "path/to/buildings.shp",
  "hidden_dim": 64,
  "num_layers": 3,
  "num_heads": 8,
  "dropout": 0.6,
  "lr": 0.005,
  "epochs": 200,
  "batch_size": 1024,
  "patience": 100,
  "device": "cuda"
}
```

Then run:

```bash
python -m src.gat --config config.json
```

## Building Features

The model extracts 12 features from building geometries:

1. **area**: Building footprint area
2. **perimeter**: Building perimeter
3. **bounds_width**: Width of bounding box
4. **bounds_height**: Height of bounding box
5. **compactness**: Circularity measure (4π·area/perimeter²)
6. **elongation**: Height/width ratio
7. **rectangularity**: Area/bounding_box_area
8. **orientation_angle**: Main axis orientation
9. **convexity**: Area/convex_hull_area
10. **num_vertices**: Number of polygon vertices
11. **perimeter_area_ratio**: Perimeter/√area
12. **centroid_distance**: Distance from district center (normalized)

## Memory Optimization

For 8GB GPU (3060ti):

- **Batch size**: 1024 nodes per batch
- **Neighbor sampling**: [15, 10] for 2-hop neighborhood
- **Node threshold**: 2000 (graphs >2000 nodes use mini-batch sampling)
- **Mixed precision**: Optional (use `--use-amp` if needed)

## Output Files

### Training
- `models/gat/best_model.pth` - Best model checkpoint
- `models/gat/final_model.pth` - Final model checkpoint
- `models/gat/training_history.json` - Training metrics history
- `runs/gat/` - TensorBoard logs

### Inference
- `output/gat/embeddings/district_{id}_embeddings.pkl` - Per-district embeddings
- `output/gat/embeddings/embeddings_summary.pkl` - Summary statistics

## Module Structure

```
src/gat/
├── __init__.py              # Module initialization
├── __main__.py              # Training entry point
├── inference.py             # Inference script
├── data/                    # Data loading
│   ├── dataset.py           # BuildingGraphDataset
│   ├── data_utils.py        # Data utilities
│   └── graph_batch_sampler.py  # Mini-batch sampling
├── models/                  # Model definitions
│   ├── gat.py              # GAT model
│   └── gat_layer.py        # GAT layer
├── training/                # Training utilities
│   ├── config.py           # Configuration
│   ├── trainer.py          # Trainer class
│   └── train_utils.py      # Training helpers
└── utils/                   # Utilities
    ├── feature_extractor.py # Feature extraction
    ├── graph_utils.py       # Graph utilities
    ├── metrics.py           # Evaluation metrics
    └── logger.py            # Logging
```

## References

- Original GAT paper: [Veličković et al. (2018)](https://arxiv.org/abs/1710.10903)
- Implementation reference: [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT)
- PyTorch Geometric: [Documentation](https://pytorch-geometric.readthedocs.io/)

