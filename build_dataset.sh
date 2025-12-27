#!/bin/bash
# Quick Start Script for Arxiv Style Transfer Dataset Construction

echo "======================================================================"
echo "ARXIV STYLE TRANSFER DATASET - QUICK START"
echo "======================================================================"

# Step 1: Build dataset (500 samples)
echo ""
echo "[STEP 1] Building dataset with 500 samples..."
echo "Command to run:"
echo "  python dataset_builder.py --num_samples 500 --output_dir /root/datasets/arxiv_style_transfer"
echo ""
echo "Note: This will take ~30-40 minutes (500 API calls with rate limiting)"
echo "      Temperature is set to 0.3 for consistent destylization"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

python dataset_builder.py \
    --num_samples 500 \
    --output_dir /root/datasets/arxiv_style_transfer

# Step 2: Validate dataset
echo ""
echo "======================================================================"
echo "[STEP 2] Validating dataset..."
echo "======================================================================"

python dataset_validator.py \
    --dataset_dir /root/datasets/arxiv_style_transfer \
    --examples 3

# Step 3: Next steps
echo ""
echo "======================================================================"
echo "CONSTRUCTION COMPLETE"
echo "======================================================================"
echo ""
echo "Dataset location: /root/datasets/arxiv_style_transfer"
echo ""
echo "Next steps:"
echo "  1. Run style transfer models on source/ texts"
echo "  2. Save outputs to transferred_<model_name>/ directories"
echo "  3. Run unified evaluation"
echo ""
echo "Quick commands:"
echo "  # View dataset structure"
echo "  tree /root/datasets/arxiv_style_transfer -L 2"
echo ""
echo "  # Check a sample"
echo "  head /root/datasets/arxiv_style_transfer/reference/ref_0000.txt"
echo "  head /root/datasets/arxiv_style_transfer/source/src_0000.txt"
echo ""
echo "  # Revalidate anytime"
echo "  python dataset_validator.py --dataset_dir /root/datasets/arxiv_style_transfer"
echo ""
