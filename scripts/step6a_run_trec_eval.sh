#!/bin/bash
set -e

# Get parameters
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/trec_eval_${TIMESTAMP}.log"
}

log "Starting TREC evaluation..."

# Check if trec_eval is installed
if ! command -v trec_eval &> /dev/null; then
    log "Error: trec_eval is not installed. Please install it first."
    exit 1
fi

# Create evaluation directory structure
RUNS_DIR="runs/evaluation/${TIMESTAMP}"
mkdir -p "${RUNS_DIR}/trec"
mkdir -p "${RUNS_DIR}/raw_metrics"
mkdir -p "${RUNS_DIR}/reports"

# Store evaluation configuration
log "Storing evaluation configuration..."
cp "${CONFIG_DIR}/evaluation_config.yaml" "${RUNS_DIR}/config.yaml"

# Run TREC evaluation for each retrieval method
for method in bm25 dense hybrid; do
    log "Evaluating $method retrieval results..."
    
    # Create method-specific directory
    mkdir -p "${RUNS_DIR}/trec/${method}"
    
    # Run standard TREC metrics
    trec_eval \
        -m map \
        -m ndcg_cut.10 \
        -m recip_rank \
        -m P.10 \
        "data/qrels/qrels.txt" \
        "runs/retrieval/${method}_${TIMESTAMP}.txt" \
        > "${RUNS_DIR}/raw_metrics/${method}_metrics.txt"
    
    # Run additional metrics if needed
    trec_eval \
        -m recall.100 \
        -m recall.1000 \
        -m precision.10 \
        -m precision.100 \
        "data/qrels/qrels.txt" \
        "runs/retrieval/${method}_${TIMESTAMP}.txt" \
        >> "${RUNS_DIR}/raw_metrics/${method}_metrics.txt"
    
    # Store a copy of the run file
    cp "runs/retrieval/${method}_${TIMESTAMP}.txt" "${RUNS_DIR}/trec/${method}/run.txt"
done

# Generate comparison report
log "Generating comparison report..."
python src/evaluation/generate_trec_report.py \
    --bm25-metrics "${RUNS_DIR}/raw_metrics/bm25_metrics.txt" \
    --dense-metrics "${RUNS_DIR}/raw_metrics/dense_metrics.txt" \
    --hybrid-metrics "${RUNS_DIR}/raw_metrics/hybrid_metrics.txt" \
    --output "${RUNS_DIR}/reports/comparison.md" \
    --log-file "${RUNS_DIR}/evaluation.log"

# Create a summary file
log "Creating evaluation summary..."
echo "TREC Evaluation Summary" > "${RUNS_DIR}/summary.txt"
echo "======================" >> "${RUNS_DIR}/summary.txt"
echo "Timestamp: ${TIMESTAMP}" >> "${RUNS_DIR}/summary.txt"
echo "" >> "${RUNS_DIR}/summary.txt"
echo "Files:" >> "${RUNS_DIR}/summary.txt"
echo "- Configuration: config.yaml" >> "${RUNS_DIR}/summary.txt"
echo "- Raw Metrics: raw_metrics/" >> "${RUNS_DIR}/summary.txt"
echo "- TREC Results: trec/" >> "${RUNS_DIR}/summary.txt"
echo "- Reports: reports/" >> "${RUNS_DIR}/summary.txt"
echo "- Log: evaluation.log" >> "${RUNS_DIR}/summary.txt"

# Create a symbolic link to the latest evaluation
ln -sfn "${RUNS_DIR}" "runs/evaluation/latest"

log "TREC evaluation completed successfully!"
log "Results stored in: ${RUNS_DIR}" 