"""Unit tests for EN-Branch modules.

Tests individual components of the bilingual pipeline:
- Language ID
- Whisper ASR
- Silero EN TTS
- Language Router
- Branch Processors
- Audio Mixer
"""

import pytest
import numpy as np
from pathlib import Path


def test_imports():
    """Test that all EN-Branch modules can be imported."""
    try:
        from glados.audio_io import SileroLanguageID
        from glados.ASR.whisper_asr import WhisperTranscriber
        from glados.TTS.tts_silero_en import SileroEnSynthesizer
        from glados.TTS.tts_silero_ru import SileroRuSynthesizer
        from glados.core.language_router import LanguageRouter, LanguageSegment
        from glados.core.branch_processor import BranchProcessor, create_branch_processors
        from glados.core.audio_mixer import AudioMixer
        print("✅ All EN-Branch modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_language_id_initialization():
    """Test Language ID model initialization."""
    try:
        from glados.audio_io import SileroLanguageID

        # Test 4-language model
        lid = SileroLanguageID(model_type="4lang", device="cpu")
        print(f"✅ Language ID initialized: {lid.languages}")

        assert lid.model_type == "4lang"
        assert lid.confidence_threshold == 0.7
        print("✅ Language ID initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Language ID test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_language_detection():
    """Test language detection on dummy audio."""
    try:
        from glados.audio_io import SileroLanguageID

        lid = SileroLanguageID(model_type="4lang", device="cpu")

        # Create dummy audio (1 second of silence)
        dummy_audio = np.zeros(16000, dtype=np.float32)

        # Detect language (will likely fail on silence but shouldn't crash)
        lang, confidence = lid.detect(dummy_audio)
        print(f"✅ Language detection result: {lang} (confidence: {confidence:.3f})")

        assert isinstance(lang, str)
        assert isinstance(confidence, float)
        print("✅ Language detection test passed")
        return True
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whisper_asr_initialization():
    """Test Whisper ASR initialization (without loading model)."""
    try:
        from glados.ASR.whisper_asr import WhisperTranscriber

        # This will load the actual model - may take time and VRAM
        print("⏳ Loading Whisper model (this may take a minute)...")
        whisper = WhisperTranscriber(model_name="tiny.en", device="cpu", fp16=False)

        print(f"✅ Whisper initialized: {whisper.model_name} on {whisper.device}")
        assert whisper.SAMPLE_RATE == 16000
        print("✅ Whisper ASR initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Whisper ASR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_silero_en_tts_initialization():
    """Test Silero EN TTS initialization."""
    try:
        from glados.TTS.tts_silero_en import SileroEnSynthesizer

        print("⏳ Loading Silero EN TTS model...")
        tts_en = SileroEnSynthesizer(speaker="en_0", sample_rate=48000, device="cpu", use_fp16=False)

        print(f"✅ Silero EN TTS initialized: speaker={tts_en.speaker}, sr={tts_en.sample_rate}")
        assert tts_en.sample_rate == 48000
        print("✅ Silero EN TTS initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Silero EN TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_silero_ru_tts_v5():
    """Test Silero RU TTS V5."""
    try:
        from glados.TTS.tts_silero_ru import SileroRuSynthesizer

        print("⏳ Loading Silero V5 RU TTS model...")
        tts_ru = SileroRuSynthesizer(speaker="xenia", sample_rate=48000, device="cpu", use_fp16=False)

        print(f"✅ Silero V5 RU TTS initialized: speaker={tts_ru.speaker}, sr={tts_ru.sample_rate}")
        assert tts_ru.sample_rate == 48000
        print("✅ Silero RU TTS V5 test passed")
        return True
    except Exception as e:
        print(f"❌ Silero RU TTS V5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test loading bilingual configuration."""
    try:
        from glados.core.engine import GladosConfig

        config_path = Path("/home/user/GLaDOS/configs/glados_bilingual_config.yaml")

        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False

        print(f"⏳ Loading configuration from {config_path}...")
        config = GladosConfig.from_yaml(str(config_path))

        print(f"✅ Config loaded successfully")
        print(f"   - enable_en_branch: {config.enable_en_branch}")
        print(f"   - asr_engine: {config.asr_engine}")
        print(f"   - voice: {config.voice}")
        print(f"   - asr_ru_engine: {config.asr_ru_engine}")
        print(f"   - asr_en_engine: {config.asr_en_engine}")
        print(f"   - voice_ru: {config.voice_ru}")
        print(f"   - voice_en: {config.voice_en}")

        assert config.enable_en_branch == True
        assert config.asr_ru_engine == "tdt"
        assert config.asr_en_engine == "whisper"
        print("✅ Configuration loading test passed")
        return True
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("EN-Branch Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("Language ID Initialization", test_language_id_initialization),
        ("Language Detection", test_language_detection),
        ("Whisper ASR Initialization", test_whisper_asr_initialization),
        ("Silero EN TTS Initialization", test_silero_en_tts_initialization),
        ("Silero RU TTS V5", test_silero_ru_tts_v5),
        ("Configuration Loading", test_config_loading),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'─' * 60}")
        print(f"Test: {name}")
        print(f"{'─' * 60}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
