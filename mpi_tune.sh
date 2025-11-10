#!/bin/bash
#BSUB -J tcb_gat_tune
#BSUB -n 64
#BSUB -W 48:00
#BSUB -M 4096
#BSUB -R "span[ptile=8]"
#BSUB -R "rusage[mem=4096]"
#BSUB -q normal
#BSUB -o logs/tuning_%J.out
#BSUB -e logs/tuning_%J.err

################################################################################
# Configuration
################################################################################

N_PROCESSES=64

ADJACENCY_DIR="/path/to/adjacency"
BUILDING_PATH="/path/to/buildings.shp"
DISTRICT_PATH="/path/to/districts.shp"
CONFIG_FILE="src/gat/training_config.yaml"
OUTPUT_DIR="experiments/hyperparameter_tuning"

################################################################################
# Run
################################################################################

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Starting hyperparameter tuning with $N_PROCESSES processes..."
echo "Output directory: $OUTPUT_DIR"

cd ~/2026tcb/neighbor_building_split
conda activate build_adj
mpirun -np "$N_PROCESSES" python -m scripts.hyperparameter_tuning \
    --config "$CONFIG_FILE" \
    --adjacency-dir "$ADJACENCY_DIR" \
    --building-path "$BUILDING_PATH" \
    --district-path "$DISTRICT_PATH" \
    --output-dir "$OUTPUT_DIR"

echo "Done!"
