#!/usr/bin/env bash

# RUNS_DIR="runs/retrieval/segment_keywords"
RUNS_DIR="runs/retrieval/sentence_keywords"

QRELS_FILE="data/qrels/10q_qrels.txt"

for run_file in "$RUNS_DIR"/*.txt; do
    if [ -f "$run_file" ]; then
        python3 src/evaluation/cal_trec.py --input_path "$run_file" --qrels_file "$QRELS_FILE"
    fi
done
