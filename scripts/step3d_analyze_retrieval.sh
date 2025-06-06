#!/bin/bash
set -e
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/retrieval_analysis_${TIMESTAMP}.log"
}
log "Analyzing retrieval results..."
python src/retrieval/analyze_results.py \
    --config "$CONFIG_DIR/retrieval_config.yaml" \
    --bm25-results "runs/retrieval/bm25_${TIMESTAMP}.txt" \
    --dense-results "runs/retrieval/dense_${TIMESTAMP}.txt" \
    --hybrid-results "runs/retrieval/hybrid_${TIMESTAMP}.txt" \
    --output "runs/retrieval/analysis_${TIMESTAMP}.json" \
    --log-file "$LOG_DIR/retrieval_analysis_${TIMESTAMP}.log"
log "Retrieval analysis completed!" 