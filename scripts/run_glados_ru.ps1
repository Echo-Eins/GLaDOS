# ============================================
# GLaDOS Russian - Full Clean & Run Script
# ============================================
#
# Use this script to completely clean all caches
# and run Russian GLaDOS with full audio pipeline
#
# Usage:
#   .\scripts\run_glados_ru.ps1
#

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GLaDOS Russian - Clean & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean Python cache
Write-Host "[1/6] Cleaning Python __pycache__..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Path . -Recurse -File -Filter "*.pyo" | Remove-Item -Force
Write-Host "      Python cache cleaned!" -ForegroundColor Green
Write-Host ""

# Step 2: Clean uv cache
Write-Host "[2/6] Cleaning uv cache..." -ForegroundColor Yellow
uv cache clean
Write-Host "      uv cache cleaned!" -ForegroundColor Green
Write-Host ""

# Step 3: Clean pip cache (optional)
Write-Host "[3/6] Cleaning pip cache..." -ForegroundColor Yellow
python -m pip cache purge 2>$null
Write-Host "      pip cache cleaned!" -ForegroundColor Green
Write-Host ""

# Step 4: Sync dependencies
Write-Host "[4/6] Syncing dependencies..." -ForegroundColor Yellow
uv sync --extra cuda --extra ru-full
Write-Host "      Dependencies synced!" -ForegroundColor Green
Write-Host ""

# Step 5: Check CUDA (optional)
Write-Host "[5/6] Checking CUDA availability..." -ForegroundColor Yellow
uv run python scripts/check_cuda.py
Write-Host ""

# Step 6: Run GLaDOS Russian
Write-Host "[6/6] Starting GLaDOS Russian..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GLaDOS Russian is starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

uv run glados start --config configs/glados_ru_config.yaml
