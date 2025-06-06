#!/bin/bash

# Exit on error
set -e

# Get parameters
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/retrieval_${TIMESTAMP}.log"
}

log "Starting retrieval step..."

# Check if indices exist
if [ ! -d "indexes/hybrid" ]; then
    log "Error: Indices not found. Please run step2_build_indices.sh first."
    exit 1
fi

# Create output directory
mkdir -p "runs/retrieval"

# Run sparse retrieval (BM25)
./scripts/step3a_run_sparse_retrieval.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Run dense retrieval
./scripts/step3b_run_dense_retrieval.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Run hybrid retrieval (fusing)
./scripts/step3c_run_hybrid_retrieval.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Analyze retrieval results
./scripts/step3d_analyze_retrieval.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

log "Retrieval step completed successfully!" 