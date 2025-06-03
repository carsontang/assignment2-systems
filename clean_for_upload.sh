#!/bin/bash
# Create a clean copy for uploading to Google Drive

echo "Creating clean copy of assignment2-systems..."

# Create temporary directory
TEMP_DIR="assignment2-systems-clean"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Copy only essential files
cp -r cs336-basics "$TEMP_DIR/"
cp -r cs336_systems "$TEMP_DIR/"
cp -r tests "$TEMP_DIR/"
cp pyproject.toml "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/" 2>/dev/null || true
cp README.md "$TEMP_DIR/"
cp CLAUDE.md "$TEMP_DIR/"
cp CS336_Assignment2_Colab.ipynb "$TEMP_DIR/" 2>/dev/null || true

# Clean Python cache from copied files
find "$TEMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$TEMP_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo "Clean copy created in: $TEMP_DIR"
echo "This folder contains only essential files and can be uploaded to Google Drive"
echo ""
echo "To create a zip for easy upload:"
echo "zip -r assignment2-systems-clean.zip $TEMP_DIR"