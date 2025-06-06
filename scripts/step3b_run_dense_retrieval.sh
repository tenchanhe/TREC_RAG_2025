#!/bin/bash
set -e
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/dense_retrieval_${TIMESTAMP}.log"
}
log "Running dense retrieval..."
python src/retrieval/run_dense_retrieval.py \
    --config "$CONFIG_DIR/retrieval_config.yaml" \
    --index "indexes/dense" \
    --output "runs/retrieval/dense_${TIMESTAMP}.txt" \
    --log-file "$LOG_DIR/dense_retrieval_${TIMESTAMP}.log"
log "Dense retrieval completed!" 