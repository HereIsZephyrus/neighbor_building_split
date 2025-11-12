#!/bin/bash
################################################################################
# Generate LSF job scripts for hyperparameter tuning
#
# This script generates individual LSF job scripts for each hyperparameter
# combination, allowing parallel execution on HPC clusters.
#
# Usage:
#   bash generate_lsf_tuning.sh [--submit]
#
# Options:
#   --submit    Automatically submit all generated jobs to LSF
################################################################################

################################################################################
# Configuration
################################################################################

# Base configuration file
BASE_CONFIG="/data/users/guxh01/2026_tcb/neighbor_building_split/src/gat/training_config.yaml"
ADJACENCY_DIR="/data/users/guxh01/2026_tcb/building_data/adjacency"
BUILDING_PATH="/data/users/guxh01/2026_tcb/building_data/training/train_building.shp"
DISTRICT_PATH="/data/users/guxh01/2026_tcb/building_data/training/train_patch.shp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/data/users/guxh01/2026_tcb/building_data/gat/experiments/hyperparameter_tuning_${TIMESTAMP}"

N_CORES=5              # Number of cores per job (should match k_fold in config)
MEMORY_PER_CORE=4096   # Memory per core in MB
TIME_LIMIT="24:00"     # Time limit per job (HH:MM)
QUEUE="normal"         # LSF queue name

################################################################################
# Parse arguments
################################################################################

SUBMIT_FLAG=""
for arg in "$@"; do
    case $arg in
        --submit)
            SUBMIT_FLAG="--submit"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--submit]"
            echo ""
            echo "Generate LSF job scripts for hyperparameter tuning."
            echo ""
            echo "Options:"
            echo "  --submit    Automatically submit all generated jobs to LSF"
            echo "  --help      Show this help message"
            echo ""
            echo "Configuration:"
            echo "  Edit this script to configure data paths and resource settings."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

python scripts/generate_tuning_jobs.py \
    --base-config "$BASE_CONFIG" \
    --adjacency-dir "$ADJACENCY_DIR" \
    --building-path "$BUILDING_PATH" \
    --district-path "$DISTRICT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-cores $N_CORES \
    --memory-per-core $MEMORY_PER_CORE \
    --time-limit "$TIME_LIMIT" \
    --queue "$QUEUE" \
    $SUBMIT_FLAG

