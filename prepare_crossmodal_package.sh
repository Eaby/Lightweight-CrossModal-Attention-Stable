#!/bin/bash

# Absolute safety: Stop if anything fails
set -e

# Navigate to your project root
cd /data/AAAI/Lightweight\ Cross-Modal\ Attention

# Name of the package
PACKAGE_NAME="Lightweight-CrossModal-Attention-Package.zip"

# Remove any existing zip
rm -f $PACKAGE_NAME

echo "🧹 Cleaning and packaging your code..."

# Build the zip excluding datasets, results, large files and .git remnants
zip -r $PACKAGE_NAME \
    Code.ipynb \
    README.md \
    .gitignore \
    automation/ \
    coco-caption/ \
    configs/ \
    data_loader.py \
    evaluation/ \
    inference/ \
    models/ \
    trainer/ \
    utils/ \
    mini_test_pipeline.py \
    -x "**/__pycache__/*" \
    -x "**/*.pyc" \
    -x "**/.git/*" \
    -x "**/.gitignore" \
    -x "datasets/*" \
    -x "results/*" \
    -x "logs/*"

echo "✅ Packaging complete."
echo "📦 Created: $PACKAGE_NAME"
