"""Integration tests for EN-Branch bilingual mode.

Tests complete system initialization with bilingual configuration.
"""

from pathlib import Path
import sys


def test_monolingual_mode():
    """Test system startup in monolingual (RU-only) mode."""
    print("\n" + "=" * 60)
    print("Test: Monolingual Mode (RU-only)")
    print("=" * 60 + "\n")

    try:
        from glados.core.engine import GladosConfig

        # Load RU config
        config_path = Path("/home/user/GLaDOS/configs/glados_ru_config.yaml")

        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False

        print(f"⏳ Loading RU configuration...")
        config = GladosConfig.from_yaml(str(config_path))

        print(f"✅ Config loaded")
        print(f"   - enable_en_branch: {config.enable_en_branch}")
        print(f"   - asr_engine: {config.asr_engine}")
        print(f"   - voice: {config.voice}")

        assert config.enable_en_branch == False, "EN-Branch should be disabled in RU config"

        print("\n⏳ Creating Glados instance (monolingual mode)...")

        # This will initialize the full system - may take time
        # We'll catch any errors during initialization
        try:
            from glados.core.engine import Glados

            glados = Glados.from_config(config)

            print(f"✅ Glados instance created successfully")
            print(f"   - enable_en_branch: {glados.enable_en_branch}")
            print(f"   - component_threads: {len(glados.component_threads)}")
            print(f"   - threads: {[t.name for t in glados.component_threads]}")

            assert glados.enable_en_branch == False
            assert len(glados.component_threads) == 4  # Standard threads only

            # Clean shutdown
            glados.shutdown_event.set()
            print("\n✅ Monolingual mode test PASSED")
            return True

        except Exception as e:
            print(f"❌ Glados initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Monolingual mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bilingual_mode_config_only():
    """Test bilingual configuration loading (without starting system)."""
    print("\n" + "=" * 60)
    print("Test: Bilingual Configuration Loading")
    print("=" * 60 + "\n")

    try:
        from glados.core.engine import GladosConfig

        config_path = Path("/home/user/GLaDOS/configs/glados_bilingual_config.yaml")

        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False

        print(f"⏳ Loading bilingual configuration...")
        config = GladosConfig.from_yaml(str(config_path))

        print(f"✅ Config loaded successfully")
        print(f"\nConfiguration details:")
        print(f"   - enable_en_branch: {config.enable_en_branch}")
        print(f"   - Main ASR: {config.asr_engine}")
        print(f"   - Main TTS: {config.voice}")
        print(f"   - RU ASR: {config.asr_ru_engine}")
        print(f"   - EN ASR: {config.asr_en_engine}")
        print(f"   - RU TTS: {config.voice_ru}")
        print(f"   - EN TTS: {config.voice_en}")
        print(f"   - LID config: {config.language_detection}")
        print(f"   - Mixer config: {config.audio_mixer}")

        # Validate config
        assert config.enable_en_branch == True
        assert config.asr_ru_engine == "tdt"
        assert config.asr_en_engine == "whisper"
        assert config.voice_ru == "silero_ru"
        assert config.voice_en == "silero_en"
        assert config.language_detection is not None
        assert config.audio_mixer is not None

        print("\n✅ Bilingual configuration test PASSED")
        return True

    except Exception as e:
        print(f"❌ Bilingual config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bilingual_mode_initialization():
    """Test full bilingual system initialization.

    WARNING: This loads ALL models (RU ASR, EN ASR, RU TTS, EN TTS, LID)
    and may require significant VRAM and time.
    """
    print("\n" + "=" * 60)
    print("Test: Bilingual Mode Initialization (FULL SYSTEM)")
    print("=" * 60 + "\n")

    print("⚠️  This test loads ALL bilingual models and may take several minutes.")
    print("⚠️  Required: ~6GB VRAM (GPU) or ~8GB RAM (CPU)")
    print("")

    try:
        from glados.core.engine import GladosConfig, Glados

        config_path = Path("/home/user/GLaDOS/configs/glados_bilingual_config.yaml")

        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False

        print(f"⏳ Loading bilingual configuration...")
        config = GladosConfig.from_yaml(str(config_path))
        print(f"✅ Config loaded")

        print(f"\n⏳ Creating Glados instance (bilingual mode)...")
        print(f"   This will load:")
        print(f"   - Main ASR: {config.asr_engine} ({config.language})")
        print(f"   - Main TTS: {config.voice}")
        print(f"   - RU ASR: {config.asr_ru_engine}")
        print(f"   - EN ASR: {config.asr_en_engine}")
        print(f"   - RU TTS: {config.voice_ru}")
        print(f"   - EN TTS: {config.voice_en}")
        print(f"   - Language ID: Silero LID")
        print("")

        try:
            glados = Glados.from_config(config)

            print(f"\n✅ Glados instance created successfully!")
            print(f"\nSystem configuration:")
            print(f"   - enable_en_branch: {glados.enable_en_branch}")
            print(f"   - component_threads: {len(glados.component_threads)}")
            print(f"   - threads: {[t.name for t in glados.component_threads]}")

            if glados.enable_en_branch:
                print(f"\nEN-Branch components:")
                print(f"   - language_router: {glados.language_router is not None}")
                print(f"   - ru_branch_processor: {glados.ru_branch_processor is not None}")
                print(f"   - en_branch_processor: {glados.en_branch_processor is not None}")
                print(f"   - audio_mixer: {glados.audio_mixer is not None}")

                assert glados.language_router is not None
                assert glados.ru_branch_processor is not None
                assert glados.en_branch_processor is not None
                assert glados.audio_mixer is not None

                # Check that we have 7 threads (4 standard + 3 EN-Branch)
                expected_threads = {
                    "SpeechListener", "LLMProcessor", "TTSSynthesizer", "AudioPlayer",
                    "RU_BranchProcessor", "EN_BranchProcessor", "AudioMixer"
                }
                actual_threads = {t.name for t in glados.component_threads}
                print(f"\nExpected threads: {expected_threads}")
                print(f"Actual threads: {actual_threads}")

                assert actual_threads == expected_threads, f"Thread mismatch: {actual_threads}"
                assert len(glados.component_threads) == 7

            # Clean shutdown
            glados.shutdown_event.set()

            print("\n✅ Bilingual mode initialization test PASSED")
            print("\n🎉 Full bilingual system initialized successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Glados initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Bilingual initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("EN-Branch Integration Tests")
    print("=" * 70)

    tests = [
        ("Bilingual Configuration Loading", test_bilingual_mode_config_only),
        ("Monolingual Mode (RU-only)", test_monolingual_mode),
        ("Bilingual Mode Initialization", test_bilingual_mode_initialization),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
