#!/bin/bash
# Install GLaDOS with CUDA support
# This script ensures PyTorch CUDA versions are installed correctly

set -e

echo "=== Installing GLaDOS with CUDA Support ==="
echo ""

# Step 1: Install base dependencies
echo "Step 1/3: Installing base dependencies..."
uv sync --extra ru-full --extra rvc

# Step 2: Force install PyTorch with CUDA 12.1
echo ""
echo "Step 2/3: Installing PyTorch with CUDA 12.1..."
uv pip install --force-reinstall \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install onnxruntime-gpu
echo ""
echo "Step 3/3: Installing onnxruntime-gpu..."
uv pip install onnxruntime-gpu>=1.16.0

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verifying CUDA installation..."
python scripts/check_cuda.py
