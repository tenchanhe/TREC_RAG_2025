#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def parse_trec_metrics(metrics_file: str) -> Dict[str, float]:
    """Parse TREC evaluation metrics from file."""
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                metric, value = line.strip().split('\t')
                metrics[metric] = float(value)
    return metrics

def generate_markdown_report(
    bm25_metrics: Dict[str, float],
    dense_metrics: Dict[str, float],
    hybrid_metrics: Dict[str, float],
    output_file: str
) -> None:
    """Generate a markdown report comparing different retrieval methods."""
    
    # Define metrics to include in the report
    metrics_to_show = [
        'map',
        'ndcg_cut_10',
        'recip_rank',
        'P_10',
        'recall_100',
        'recall_1000',
        'precision_10',
        'precision_100'
    ]
    
    # Generate markdown table
    markdown = "# TREC Evaluation Results\n\n"
    markdown += "| Metric | BM25 | Dense | Hybrid |\n"
    markdown += "|--------|------|--------|--------|\n"
    
    for metric in metrics_to_show:
        bm25_value = bm25_metrics.get(metric, 'N/A')
        dense_value = dense_metrics.get(metric, 'N/A')
        hybrid_value = hybrid_metrics.get(metric, 'N/A')
        
        markdown += f"| {metric} | {bm25_value:.4f} | {dense_value:.4f} | {hybrid_value:.4f} |\n"
    
    # Add summary section
    markdown += "\n## Summary\n\n"
    markdown += "### Best Performance by Metric\n\n"
    
    for metric in metrics_to_show:
        values = {
            'BM25': bm25_metrics.get(metric, 0),
            'Dense': dense_metrics.get(metric, 0),
            'Hybrid': hybrid_metrics.get(metric, 0)
        }
        best_method = max(values.items(), key=lambda x: x[1])[0]
        markdown += f"- {metric}: {best_method}\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(markdown)

def main():
    parser = argparse.ArgumentParser(description='Generate TREC evaluation report')
    parser.add_argument('--bm25-metrics', required=True, help='Path to BM25 metrics file')
    parser.add_argument('--dense-metrics', required=True, help='Path to dense metrics file')
    parser.add_argument('--hybrid-metrics', required=True, help='Path to hybrid metrics file')
    parser.add_argument('--output', required=True, help='Path to output markdown file')
    parser.add_argument('--log-file', required=True, help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting TREC report generation")
    
    try:
        # Parse metrics
        bm25_metrics = parse_trec_metrics(args.bm25_metrics)
        dense_metrics = parse_trec_metrics(args.dense_metrics)
        hybrid_metrics = parse_trec_metrics(args.hybrid_metrics)
        
        # Generate report
        generate_markdown_report(
            bm25_metrics,
            dense_metrics,
            hybrid_metrics,
            args.output
        )
        
        logger.info(f"Report generated successfully: {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main() 