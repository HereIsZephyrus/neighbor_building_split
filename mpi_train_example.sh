#!/bin/bash

# Shell script to launch MPI-parallel k-fold cross-validation GAT model training

source .venv/bin/activate
mpirun -n 8 python -m src.gat \
    --train \
    --mode cv \
    --model-identifier <MODEL_VERSION> \
    --adjacency-dir <ADJACENCY_DIR> \
    --sample-buildings <TRAIN_BUILDING_SHP> \
    --sample-districts <TRAIN_PATCH_SHP> \
    --output-root-dir <OUTPUT_ROOT_DIR> \
    --config <TRAINING_CONFIG_YAML>
