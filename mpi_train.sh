#!/bin/bash

# Shell script to launch MPI-parallel k-fold cross-validation GAT model training

source .venv/bin/activate
mpirun -n 8 python -m src.gat \
    --train \
    --mode cv \
    --model-identifier vk5 \
    --adjacency-dir /mnt/warehouse/neighborhood2/output2/voronoi_diagrams \
    --sample-buildings /mnt/repo/wuhanthermal/TAZ/training/train_building.shp \
    --sample-districts /mnt/repo/wuhanthermal/TAZ/training/train_patch.shp \
    --output-root-dir /mnt/warehouse/gat \
    --config src/gat/training_config.yaml
