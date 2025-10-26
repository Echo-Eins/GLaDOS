# RVC Voice Conversion Setup Guide

This guide explains how to set up and use RVC (Retrieval-based Voice Conversion) for GLaDOS voice synthesis.

## Overview

The GLaDOS Russian TTS pipeline now includes full RVC voice conversion support:

1. **Silero TTS** - Generates base Russian speech
2. **RVC Voice Conversion** - Converts to GLaDOS voice using inferrvc
3. **Audio Processing** - Applies EQ, compression, and reverb

## Installation

### Basic Installation (without RVC)

Install the Russian TTS package without RVC (uses simple pitch shifting):

```bash
# Using uv (recommended)
uv pip install -e ".[ru-full,cuda]"

# Or using pip
pip install -e ".[ru-full,cuda]"
```

### Full RVC Installation (Python 3.12 supported!)

**🎉 Good News:** Full RVC now works with **Python 3.12**!

We use `inferrvc` (CircuitCM/RVC-inference), which supports Python 3.8-3.12.

#### Install RVC (Python 3.11 or 3.12)

```bash
# Using uv (recommended)
uv pip install -e ".[ru-full,rvc,cuda]"

# Or using pip
pip install -e ".[ru-full,rvc,cuda]"
```

This will install:
- `inferrvc` - High-performance RVC inference library
- `faiss-cpu>=1.8.0` - Feature index search (Python 3.12 compatible!)
- `fairseq` fork - For Python 3.11+ compatibility (auto-installed)

### What's Different?

**Old setup (rvc-python):**
- ❌ Only Python 3.11 (faiss-cpu 1.7.3 limitation)
- ❌ Complex API with temporary files
- ❌ Slower inference

**New setup (inferrvc):**
- ✅ Python 3.8-3.12 support
- ✅ Direct torch.Tensor processing
- ✅ Optimized for performance
- ✅ Includes RMVPE (best pitch estimator)
- ✅ Fewer dependencies

RVC dependencies (installed with `[rvc]`):
- `inferrvc` - High-performance RVC inference
- `faiss-cpu>=1.8.0` - Feature search (Python 3.12 compatible!)
- `fairseq` fork - For Python 3.11+ (auto-installed)

Standard dependencies (installed with `ru-full`):
- `librosa` - Audio processing
- `pyworld` - F0 extraction
- `praat-parselmouth` - Pitch analysis
- `resampy` - Audio resampling
- `torchcrepe` - Deep learning pitch detection
- `rvc-python` - RVC inference library

### GPU Support

For CUDA support, ensure you have the appropriate PyTorch version:

```bash
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Basic Usage

```python
from glados.TTS.tts_glados_ru import GLaDOSRuSynthesizer

# Initialize synthesizer with RVC enabled
synth = GLaDOSRuSynthesizer(
    enable_rvc=True,
    enable_audio_processing=True,
    device='cuda'  # or 'cpu'
)

# Generate speech
audio = synth.generate_speech_audio("Привет, это GLaDOS")
```

### RVC Parameters

You can tune RVC parameters for better results:

```python
from glados.audio_processing.rvc_processor import create_rvc_processor

rvc = create_rvc_processor(
    model_path="models/TTS/GLaDOS_ru/ru_glados.pth",
    index_path="models/TTS/GLaDOS_ru/added_IVF424_Flat_nprobe_1_ru_glados_v2.index",
    simple_mode=False,
    device='cuda',

    # RVC parameters
    f0_method="harvest",      # F0 extraction: 'harvest', 'pm', 'crepe', 'rmvpe'
    f0_up_key=0,             # Pitch shift in semitones (-12 to +12)
    index_rate=0.75,         # Feature index influence (0.0 to 1.0)
    filter_radius=3,         # Median filter for F0 smoothing (0-7)
    rms_mix_rate=0.25,       # RMS envelope mixing (0.0 to 1.0)
    protect=0.33,            # Consonant protection (0.0 to 0.5)
)

# Process audio
converted = rvc.process(audio, input_sample_rate=48000)
```

### F0 Extraction Methods

- **rmvpe** - Best quality, good performance (default, recommended)
- **harvest** - Robust, slower but accurate
- **pm** - Parselmouth, fast and good for real-time
- **crepe** - Deep learning based, very accurate but slow
- **dio** - Fast but lower quality

### Parameter Tuning Guide

#### f0_up_key (Pitch Shift)
- Range: -12 to +12 semitones
- 0 = no change
- Positive values = higher pitch
- Negative values = lower pitch
- Typical GLaDOS: 0 to -2

#### index_rate (Feature Retrieval)
- Range: 0.0 to 1.0
- Higher = more training data influence
- Lower = more original voice preserved
- Recommended: 0.7 to 0.8

#### filter_radius (F0 Smoothing)
- Range: 0 to 7
- Higher = smoother pitch contour
- Lower = more natural variations
- Recommended: 3

#### rms_mix_rate (Loudness Matching)
- Range: 0.0 to 1.0
- Controls how much output loudness matches input
- Recommended: 0.2 to 0.3

#### protect (Consonant Protection)
- Range: 0.0 to 0.5
- Protects voiceless consonants from excessive conversion
- Higher = more protection
- Recommended: 0.33

## Performance

### Processing Pipeline

The complete pipeline processes audio as follows:

1. **Silero TTS**: ~0.5-1.0s for typical sentence
2. **RVC Conversion**: ~1.5-3.0s depending on length and method
3. **Audio Processing**: ~0.1-0.3s

Total RTF (Real-Time Factor) is typically 1.5-2.5x on GPU.

### Optimization Tips

1. **Use GPU**: RVC is much faster on CUDA
2. **Choose faster F0 method**: 'pm' or 'harvest' over 'crepe'
3. **Disable index**: Set `index_rate=0` if quality is acceptable
4. **Batch processing**: Process multiple sentences together

## Fallback Mode (Without RVC)

If `inferrvc` is not installed, the system **automatically** falls back to `SimpleRVCProcessor`:

- ✅ **Basic pitch shifting** - Adjusts pitch without full voice conversion
- ✅ **Fast processing** - Much faster than full RVC
- ✅ **No additional dependencies** - Works immediately
- ✅ **Good for testing** - Quick iteration on TTS

**What you get without RVC:**
- Silero TTS generates natural Russian speech
- SimpleRVCProcessor applies pitch shifting (configurable via `f0_up_key`)
- Audio processing (EQ, compression, reverb) is still applied
- Result sounds good but won't have full GLaDOS voice conversion

**What you get with full RVC (inferrvc):**
- ✨ Full voice conversion to GLaDOS timbre
- ✨ Feature retrieval for better quality
- ✨ RMVPE pitch extraction (best quality)
- ✨ More authentic GLaDOS sound
- ✨ Works on Python 3.12!

## Troubleshooting

### "inferrvc not available"
Install RVC dependencies:
```bash
uv pip install -e ".[rvc]"
```

### "FAISS not available"
This means faiss-cpu is not installed. Install with:
```bash
pip install "faiss-cpu>=1.8.0"
```

### "fairseq" installation fails on Python 3.11+
This should install automatically, but if it fails:
```bash
pip install https://github.com/One-sixth/fairseq/archive/main.zip
```

### "pyworld not available"
Install with: `pip install pyworld` - needed for F0 extraction

### RVC works but quality is poor
Try adjusting parameters:
- Increase `index_rate` (0.7 → 0.85)
- Try different `f0_method` ('rmvpe' is usually best)
- Adjust `protect` (0.33 → 0.4)

### RVC output sounds wrong
- Check model version (v1 vs v2)
- Adjust `index_rate` (try 0.5-0.9)
- Try different `f0_method`
- Adjust `protect` parameter

### Slow performance
- Use GPU instead of CPU
- Switch to 'pm' or 'harvest' F0 method
- Reduce `filter_radius`

## Models

The GLaDOS Russian model includes:
- **ru_glados.pth** - RVC model weights (52MB)
- **added_IVF424_Flat_nprobe_1_ru_glados_v2.index** - Feature index (52MB)

These are located in `models/TTS/GLaDOS_ru/`

## Real-time Processing

For future real-time voice conversion, consider:
1. Streaming audio in small chunks
2. Using faster F0 methods (pm, rmvpe)
3. Reducing buffer sizes
4. GPU acceleration

Implementation of real-time RVC is planned for future updates.

## References

- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [rvc-python](https://github.com/daswer123/rvc-python)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
