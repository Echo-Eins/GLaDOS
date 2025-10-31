#!/usr/bin/env python3
"""Test configuration parsing without loading heavy dependencies."""

import yaml
from pathlib import Path


def test_config_parsing():
    """Test that bilingual config can be parsed."""
    print("\n" + "=" * 60)
    print("Configuration Parsing Test")
    print("=" * 60 + "\n")

    config_path = Path("/home/user/GLaDOS/configs/glados_bilingual_config.yaml")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    print(f"✅ Config file exists: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        print("\n✅ YAML parsed successfully")

        glados_config = data.get("Glados", {})

        print("\nConfiguration fields:")
        print(f"   - llm_model: {glados_config.get('llm_model')}")
        print(f"   - audio_io: {glados_config.get('audio_io')}")
        print(f"   - asr_engine: {glados_config.get('asr_engine')}")
        print(f"   - voice: {glados_config.get('voice')}")
        print(f"   - language: {glados_config.get('language')}")
        print(f"\nEN-Branch fields:")
        print(f"   - enable_en_branch: {glados_config.get('enable_en_branch')}")
        print(f"   - asr_ru_engine: {glados_config.get('asr_ru_engine')}")
        print(f"   - asr_en_engine: {glados_config.get('asr_en_engine')}")
        print(f"   - voice_ru: {glados_config.get('voice_ru')}")
        print(f"   - voice_en: {glados_config.get('voice_en')}")
        print(f"   - language_detection: {glados_config.get('language_detection')}")
        print(f"   - audio_mixer: {glados_config.get('audio_mixer')}")

        # Validate
        assert glados_config.get("enable_en_branch") == True
        assert glados_config.get("asr_ru_engine") == "tdt"
        assert glados_config.get("asr_en_engine") == "whisper"
        assert glados_config.get("voice_ru") == "silero_ru"
        assert glados_config.get("voice_en") == "silero_en"
        assert glados_config.get("language_detection") is not None
        assert glados_config.get("audio_mixer") is not None

        print("\n✅ All EN-Branch fields present and valid!")

        # Check that legacy fields are also present
        assert glados_config.get("asr_engine") == "tdt"
        assert glados_config.get("voice") == "glados_ru"
        assert glados_config.get("language") == "ru"

        print("✅ Legacy fields present for backward compatibility!")

        print("\n" + "=" * 60)
        print("✅ Configuration parsing test PASSED")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_config_parsing()
    sys.exit(0 if success else 1)
