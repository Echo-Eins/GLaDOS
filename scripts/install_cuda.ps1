# Install GLaDOS with CUDA support (Windows PowerShell)
# This script ensures PyTorch CUDA versions are installed correctly

$ErrorActionPreference = "Stop"

Write-Host "=== Installing GLaDOS with CUDA Support ===" -ForegroundColor Green
Write-Host ""

# Step 1: Install base dependencies
Write-Host "Step 1/3: Installing base dependencies..." -ForegroundColor Cyan
uv sync --extra ru-full --extra rvc

# Step 2: Force install PyTorch with CUDA 12.1
Write-Host ""
Write-Host "Step 2/3: Installing PyTorch with CUDA 12.1..." -ForegroundColor Cyan
uv pip install --force-reinstall `
    torch==2.4.0 `
    torchaudio==2.4.0 `
    torchvision==0.19.0 `
    --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install onnxruntime-gpu
Write-Host ""
Write-Host "Step 3/3: Installing onnxruntime-gpu..." -ForegroundColor Cyan
uv pip install onnxruntime-gpu

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Verifying CUDA installation..." -ForegroundColor Yellow
python scripts/check_cuda.py
