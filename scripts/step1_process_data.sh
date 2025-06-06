#!/bin/bash

# Exit on error
set -e

# Get parameters
TIMESTAMP=$1
CONFIG_DIR=$2
LOG_DIR=$3

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/data_processing_${TIMESTAMP}.log"
}

log "Starting data processing step..."

# Check if data exists
if [ ! -d "data/corpus/raw" ]; then
    log "Error: Raw data not found. Please run download.sh first."
    exit 1
fi

# Run data processing
log "Running corpus processing..."
python src/data/process_corpus.py \
    --config "$CONFIG_DIR/data_config.yaml" \
    --input "data/corpus/raw" \
    --output "data/corpus/processed" \
    --log-file "$LOG_DIR/data_processing_${TIMESTAMP}.log"

# Validate processed data
log "Validating processed data..."
python src/data/validate_processed.py \
    --config "$CONFIG_DIR/data_config.yaml" \
    --input "data/corpus/processed" \
    --output "data/corpus/validation/validation_${TIMESTAMP}.json" \
    --log-file "$LOG_DIR/data_validation_${TIMESTAMP}.log"

log "Data processing step completed successfully!" 