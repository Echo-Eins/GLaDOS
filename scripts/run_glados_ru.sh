#!/bin/bash
# ============================================
# GLaDOS Russian - Full Clean & Run Script
# ============================================
#
# Use this script to completely clean all caches
# and run Russian GLaDOS with full audio pipeline
#
# Usage:
#   ./scripts/run_glados_ru.sh
#

echo "========================================"
echo "GLaDOS Russian - Clean & Run"
echo "========================================"
echo ""

# Step 1: Clean Python cache
echo "[1/6] Cleaning Python __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "      ✓ Python cache cleaned!"
echo ""

# Step 2: Clean uv cache
echo "[2/6] Cleaning uv cache..."
uv cache clean
echo "      ✓ uv cache cleaned!"
echo ""

# Step 3: Clean pip cache (optional)
echo "[3/6] Cleaning pip cache..."
python -m pip cache purge 2>/dev/null || true
echo "      ✓ pip cache cleaned!"
echo ""

# Step 4: Sync dependencies
echo "[4/6] Syncing dependencies..."
uv sync --extra cuda --extra ru-full
echo "      ✓ Dependencies synced!"
echo ""

# Step 5: Check CUDA (optional)
echo "[5/6] Checking CUDA availability..."
uv run python scripts/check_cuda.py
echo ""

# Step 6: Run GLaDOS Russian
echo "[6/6] Starting GLaDOS Russian..."
echo ""
echo "========================================"
echo "GLaDOS Russian is starting..."
echo "========================================"
echo ""

uv run glados start --config configs/glados_ru_config.yaml
