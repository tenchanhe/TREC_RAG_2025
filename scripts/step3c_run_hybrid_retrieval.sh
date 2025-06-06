#!/bin/bash
set -e
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/hybrid_retrieval_${TIMESTAMP}.log"
}
log "Running hybrid retrieval (fusing)..."
python src/retrieval/run_hybrid_retrieval.py \
    --config "$CONFIG_DIR/retrieval_config.yaml" \
    --bm25-results "runs/retrieval/bm25_${TIMESTAMP}.txt" \
    --dense-results "runs/retrieval/dense_${TIMESTAMP}.txt" \
    --output "runs/retrieval/hybrid_${TIMESTAMP}.txt" \
    --log-file "$LOG_DIR/hybrid_retrieval_${TIMESTAMP}.log"
log "Hybrid retrieval completed!" 