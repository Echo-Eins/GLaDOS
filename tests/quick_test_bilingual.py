#!/usr/bin/env python3
"""Quick test script for bilingual mode.

This script performs a quick sanity check without loading heavy models.
"""

import sys
from pathlib import Path


def quick_test():
    """Quick sanity check for bilingual mode."""
    print("\n" + "=" * 60)
    print("Quick Bilingual Mode Test")
    print("=" * 60 + "\n")

    # Step 1: Check imports
    print("Step 1: Checking imports...")
    try:
        from glados.core.engine import GladosConfig, Glados
        from glados.audio_io import SileroLanguageID
        from glados.ASR.whisper_asr import WhisperTranscriber
        from glados.TTS.tts_silero_en import SileroEnSynthesizer
        from glados.TTS.tts_silero_ru import SileroRuSynthesizer
        from glados.core.language_router import LanguageRouter
        from glados.core.branch_processor import BranchProcessor
        from glados.core.audio_mixer import AudioMixer
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Step 2: Check config file exists
    print("\nStep 2: Checking configuration file...")
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    config_path = project_root / "configs" / "glados_bilingual_config.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    print(f"✅ Config file found: {config_path}")

    # Step 3: Load configuration
    print("\nStep 3: Loading configuration...")
    try:
        config = GladosConfig.from_yaml(str(config_path))
        print("✅ Configuration loaded successfully")
        print(f"   - enable_en_branch: {config.enable_en_branch}")
        print(f"   - asr_ru_engine: {config.asr_ru_engine}")
        print(f"   - asr_en_engine: {config.asr_en_engine}")
        print(f"   - voice_ru: {config.voice_ru}")
        print(f"   - voice_en: {config.voice_en}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Validate configuration
    print("\nStep 4: Validating configuration...")
    try:
        assert config.enable_en_branch == True, "EN-Branch not enabled"
        assert config.asr_ru_engine == "tdt", f"Wrong RU ASR: {config.asr_ru_engine}"
        assert config.asr_en_engine == "whisper", f"Wrong EN ASR: {config.asr_en_engine}"
        assert config.voice_ru == "silero_ru", f"Wrong RU voice: {config.voice_ru}"
        assert config.voice_en == "silero_en", f"Wrong EN voice: {config.voice_en}"
        print("✅ Configuration valid")
    except AssertionError as e:
        print(f"❌ Config validation failed: {e}")
        return False

    # Step 5: Check that models can be instantiated (without loading)
    print("\nStep 5: Checking model classes...")
    try:
        # Just check that classes exist and can be imported
        from glados.ASR import get_audio_transcriber
        from glados.TTS import get_speech_synthesizer

        print("✅ ASR factory function available")
        print("✅ TTS factory function available")
    except Exception as e:
        print(f"❌ Model factory check failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ Quick test PASSED!")
    print("=" * 60)
    print("\nThe bilingual system configuration is valid and ready.")
    print("To start the full system, run:")
    print("  uv run glados start --config configs/glados_bilingual_config.yaml")
    print("")

    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
