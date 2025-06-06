#!/bin/bash

# Exit on error
set -e

# Configuration
CORPUS_URL="https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2.1_doc.tar"
SEGMENTED_CORPUS_URL="https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar"
CORPUS_MD5="a5950665d6448d3dbaf7135645f1e074"
SEGMENTED_CORPUS_MD5="3799e7611efffd8daeb257e9ccca4d60"

# Create necessary directories
mkdir -p data/corpus/raw
mkdir -p data/corpus/processed
mkdir -p data/corpus/validation

# Function to download and verify
download_and_verify() {
    local url=$1
    local output_file=$2
    local expected_md5=$3
    
    echo "Downloading $output_file..."
    wget -q --show-progress -O "$output_file" "$url"
    
    echo "Verifying MD5 checksum..."
    local actual_md5=$(md5sum "$output_file" | cut -d' ' -f1)
    
    if [ "$actual_md5" = "$expected_md5" ]; then
        echo "MD5 verification successful!"
    else
        echo "MD5 verification failed!"
        echo "Expected: $expected_md5"
        echo "Got: $actual_md5"
        exit 1
    fi
}

# Download and verify corpus
download_and_verify "$CORPUS_URL" "data/corpus/raw/msmarco_v2.1_doc.tar" "$CORPUS_MD5"
download_and_verify "$SEGMENTED_CORPUS_URL" "data/corpus/raw/msmarco_v2.1_doc_segmented.tar" "$SEGMENTED_CORPUS_MD5"

# Extract files
echo "Extracting corpus files..."
tar -xf "data/corpus/raw/msmarco_v2.1_doc.tar" -C "data/corpus/raw"
tar -xf "data/corpus/raw/msmarco_v2.1_doc_segmented.tar" -C "data/corpus/raw"

# Create validation report
echo "Creating validation report..."
echo "Download completed successfully at $(date)" > "data/corpus/validation/download_report.txt"
echo "Files downloaded:" >> "data/corpus/validation/download_report.txt"
echo "- msmarco_v2.1_doc.tar" >> "data/corpus/validation/download_report.txt"
echo "- msmarco_v2.1_doc_segmented.tar" >> "data/corpus/validation/download_report.txt"

echo "Download and extraction completed successfully!" 