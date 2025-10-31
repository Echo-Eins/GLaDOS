#!/usr/bin/env python3
"""Quick test to verify Silero V5 Russian TTS loading."""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_silero_v5_loading():
    """Test that Silero V5 model loads correctly with new download method."""

    print("\n" + "="*60)
    print("Silero V5 Russian TTS Loading Test")
    print("="*60 + "\n")

    try:
        from glados.TTS.tts_silero_ru import SileroRuSynthesizer

        print("✅ SileroRuSynthesizer imported successfully")

        # Initialize synthesizer (will download V5 model if not cached)
        print("\n📦 Initializing Silero V5 TTS (may download model on first run)...")
        tts = SileroRuSynthesizer(speaker="xenia", device="cpu")

        print("✅ Silero V5 model loaded successfully!")
        print(f"   - Speaker: {tts.speaker}")
        print(f"   - Sample rate: {tts.sample_rate} Hz")
        print(f"   - Device: {tts.device}")
        print(f"   - FP16: {tts.use_fp16}")

        # Test speech generation
        print("\n🎤 Testing speech generation...")
        test_text = "Привет! Это тест синтеза речи."
        audio = tts.generate_speech_audio(test_text)

        print(f"✅ Generated {len(audio)} audio samples ({len(audio)/tts.sample_rate:.2f}s)")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio dtype: {audio.dtype}")

        print("\n" + "="*60)
        print("✅ All tests PASSED!")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Test FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        return False

if __name__ == "__main__":
    success = test_silero_v5_loading()
    sys.exit(0 if success else 1)
