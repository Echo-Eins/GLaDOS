#!/usr/bin/env python3
"""Check CUDA availability and GPU information."""

import sys


def check_cuda():
    """Check CUDA availability and print GPU information."""
    print("=== CUDA Availability Check ===\n")

    # Check PyTorch CUDA
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")

        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"    Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

                # Get device properties
                props = torch.cuda.get_device_properties(i)
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("\n  No CUDA devices found. Reasons could be:")
            print("  - NVIDIA GPU not present")
            print("  - CUDA drivers not installed")
            print("  - PyTorch CPU-only version installed")

    except ImportError:
        print("✗ PyTorch not installed")
        return False

    # Check ONNX Runtime GPU
    print("\n=== ONNX Runtime GPU Check ===\n")
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime installed: {ort.__version__}")

        providers = ort.get_available_providers()
        print(f"  Available providers: {providers}")

        if 'CUDAExecutionProvider' in providers:
            print("  ✓ CUDA Execution Provider available")
        else:
            print("  ✗ CUDA Execution Provider NOT available")
            print("    Install: pip install onnxruntime-gpu")

    except ImportError:
        print("✗ ONNX Runtime not installed")

    # Quick test
    print("\n=== Quick CUDA Test ===\n")
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ CUDA tensor operations working correctly")
            del x, y, z
            torch.cuda.empty_cache()
        else:
            print("⚠ Skipping CUDA test (no GPU available)")
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")

    print("\n=== Recommendation ===\n")
    if cuda_available:
        print("✓ CUDA is available and working!")
        print("  Use: GLaDOSRuSynthesizer(device='cuda')")
    else:
        print("⚠ CUDA not available, using CPU")
        print("  Use: GLaDOSRuSynthesizer(device='cpu')")


if __name__ == "__main__":
    check_cuda()
