#!/usr/bin/env python3
"""Quick test to verify Silero LID loads correctly from silero-vad repo."""

from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_silero_lid_loading():
    """Test that Silero LID model loads correctly from silero-vad."""

    print("\n" + "="*60)
    print("Silero Language ID Loading Test")
    print("="*60 + "\n")

    try:
        from glados.audio_io.language_id import SileroLanguageID

        print("✅ SileroLanguageID imported successfully")

        # Initialize LID model
        print("\n📦 Initializing Silero LID (may take a moment on first run)...")
        lid = SileroLanguageID(model_type="95lang", device="cpu")

        print("✅ Silero LID model loaded successfully!")
        print(f"   - Model type: {lid.model_type}")
        print(f"   - Device: {lid.device}")
        print(f"   - Confidence threshold: {lid.confidence_threshold}")
        print(f"   - Supported languages: 95")

        # Test detection with dummy audio
        print("\n🎤 Testing language detection with dummy audio...")

        # Create 1 second of random noise
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        lang_code, confidence = lid.detect(audio)

        print(f"✅ Detection completed!")
        print(f"   - Detected language: {lang_code}")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Note: Random noise may give low-confidence results")

        print("\n" + "="*60)
        print("✅ All tests PASSED!")
        print("="*60 + "\n")

        print("ℹ️  Note: The LID model is now loaded from 'snakers4/silero-vad'")
        print("   This is the correct repo for Silero Language Identification.")

        return True

    except Exception as e:
        print(f"\n❌ Test FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        return False

if __name__ == "__main__":
    success = test_silero_lid_loading()
    sys.exit(0 if success else 1)
