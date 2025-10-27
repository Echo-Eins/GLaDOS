# Installing GLaDOS with CUDA Support

This guide explains how to properly install GLaDOS with CUDA acceleration for maximum performance.

## Prerequisites

- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
- **NVIDIA CUDA Drivers** installed (version 12.1 or compatible)
- **Python 3.12** or higher
- **uv** package manager

Check your GPU and CUDA version:
```bash
nvidia-smi
```

## Installation Methods

### Method 1: Automated Installation (Recommended)

#### Windows (PowerShell):
```powershell
.\scripts\install_cuda.ps1
```

#### Linux/Mac (Bash):
```bash
chmod +x scripts/install_cuda.sh
./scripts/install_cuda.sh
```

### Method 2: Manual Installation

#### Step 1: Install base dependencies
```bash
uv sync --extra ru-full --extra rvc
```

#### Step 2: Install PyTorch with CUDA 12.1
```bash
uv pip install --force-reinstall \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install onnxruntime-gpu
```bash
uv pip install onnxruntime-gpu
```

## Verification

After installation, verify CUDA is working:

```bash
python scripts/check_cuda.py
```

**Expected output (successful):**
```
=== CUDA Availability Check ===

✓ PyTorch installed: 2.4.0+cu121
✓ CUDA available: True
  CUDA version: 12.1
  Device 0: NVIDIA GeForce RTX 3060
  Memory: 12288 MB
```

**Common issues:**

If you see `PyTorch installed: 2.4.0+cpu`:
- PyTorch CPU-only version is installed
- Re-run Step 2 with `--force-reinstall`

If you see `CUDA available: False`:
- GPU drivers not installed or outdated
- Incompatible CUDA version
- Try CUDA 11.8 instead: change `cu121` to `cu118` in install command

## Alternative CUDA Versions

### CUDA 11.8 (for older GPUs):
```bash
uv pip install --force-reinstall \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.4 (latest):
```bash
uv pip install --force-reinstall \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

## Performance Impact

With CUDA + FP16 enabled:

| Component | CPU (FP32) | CUDA (FP16) | Speedup |
|-----------|-----------|-------------|---------|
| **Silero TTS** | 1.5s | 0.6s | **2.5x** |
| **RVC Processing** | 12s | 2.4s | **5x** |
| **Total (3 paragraphs)** | 50s | 12s | **4x** |

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU for some components

### Issue: "No module named 'torch'"
**Solution:** Reinstall PyTorch with Step 2

### Issue: "RuntimeError: CUDA error: device-side assert triggered"
**Solution:** Update GPU drivers or try CUDA 11.8

### Issue: Still showing CPU version after install
**Solution:**
```bash
# Completely remove torch
uv pip uninstall torch torchaudio torchvision

# Reinstall with force
uv pip install --force-reinstall \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

## CPU-Only Installation

If you don't have an NVIDIA GPU, use:

```bash
uv sync --extra ru-full --extra rvc --extra cpu
```

This will install CPU-only versions (slower but works on any hardware).
