#!/bin/bash

# Exit on error
set -e

# Configuration
CONFIG_DIR="configs"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "runs/retrieval"
mkdir -p "runs/generation"
mkdir -p "runs/evaluation"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log "Error: $1 is not installed"
        exit 1
    fi
}

# Check required commands
check_command python
check_command pip

# # Check if virtual environment exists
# if [ ! -d ".venv" ]; then
#     log "Creating virtual environment..."
#     python -m venv .venv
# fi

# # Activate virtual environment
# log "Activating virtual environment..."
# source .venv/bin/activate

# # Install/update dependencies
# log "Installing dependencies..."
# pip install -r requirements.txt

# Run each step
log "Starting pipeline execution..."

# # Step 1: Data Processing
# log "Step 1: Data Processing"
# ./scripts/step1_process_data.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Step 2: Build Indices
log "Step 2: Building Indices"
./scripts/step2_build_indices.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Step 3: Run Retrieval (Track R)
log "Step 3: Running Retrieval Task"
./scripts/step3_run_retrieval.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Step 4: Run Generation (Track AG)
log "Step 4: Running Generation Task"
./scripts/step4_run_generation.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Step 5: Run RAG (Track RAG)
log "Step 5: Running RAG Task"
./scripts/step5_run_rag.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# Step 6: Run Evaluation
log "Step 6: Running Evaluation"
./scripts/step6_run_evaluation.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

# # Step 7: Generate Report
# log "Step 7: Generating Final Report"
# ./scripts/step7_generate_report.sh "$TIMESTAMP" "$CONFIG_DIR" "$LOG_DIR"

log "Pipeline completed successfully!"
log "Results are available in runs/evaluation/report_${TIMESTAMP}.md" 