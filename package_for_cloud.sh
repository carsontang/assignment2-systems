#!/bin/bash
set -euo pipefail

# Script to package the assignment for cloud GPU deployment
echo "Packaging CS336 Assignment 2 for cloud deployment..."

# Set output filename
OUTPUT_FILE="cs336_assignment2_cloud.tar.gz"

# Remove old package if exists
rm -f "$OUTPUT_FILE"

# Create the tarball with necessary files
tar -czf "$OUTPUT_FILE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='*.log' \
    --exclude='wandb' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.bin' \
    --exclude='*.ckpt' \
    --exclude='test_results.xml' \
    --exclude='cs336-spring2024-assignment-2-submission.zip' \
    ./cs336-basics \
    ./cs336_systems \
    ./tests \
    ./pyproject.toml \
    ./requirements.txt \
    ./setup_cloud.sh \
    ./README.md \
    ./CLAUDE.md

echo "Package created: $OUTPUT_FILE"
echo "Size: $(du -h $OUTPUT_FILE | cut -f1)"
echo ""
echo "To deploy on cloud GPU:"
echo "1. Upload $OUTPUT_FILE to your cloud instance"
echo "2. Extract with: tar -xzf $OUTPUT_FILE"
echo "3. Run: ./setup_cloud.sh"