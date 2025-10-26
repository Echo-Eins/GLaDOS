#!/usr/bin/env python3
"""Example usage of GLaDOS Russian voice synthesis pipeline.

This example demonstrates:
1. Basic usage of GLaDOS Russian TTS
2. Audio processing pipeline configuration
3. Preset management
4. RVC voice conversion

Requirements:
    pip install torch silero librosa scipy faiss-cpu
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glados.TTS.tts_glados_ru import GLaDOSRuSynthesizer
from glados.audio_processing import (
    AudioProcessingPipeline,
    PresetManager,
    run_audio_processor_tui,
)
import soundfile as sf
from loguru import logger


def basic_usage():
    """Example 1: Basic usage of GLaDOS Russian TTS."""
    logger.info("=== Example 1: Basic Usage ===")

    # Initialize synthesizer
    glados = GLaDOSRuSynthesizer()

    # Generate speech
    text = "Привет! Я GLaDOS. Добро пожаловать в лабораторию тестирования."
    audio = glados.generate_speech_audio(text)

    # Save to file
    output_path = Path("output_basic.wav")
    sf.write(output_path, audio, glados.sample_rate)
    logger.info(f"Audio saved to {output_path}")


def with_custom_preset():
    """Example 2: Using custom audio processing preset."""
    logger.info("=== Example 2: Custom Preset ===")

    # Initialize preset manager and create custom preset
    preset_manager = PresetManager()

    custom_config = {
        "eq": [
            {"type": "highpass", "frequency": 100, "gain_db": 0, "q_factor": 0.7},
            {"type": "peak", "frequency": 2000, "gain_db": 4.0, "q_factor": 1.5},
            {"type": "highshelf", "frequency": 6000, "gain_db": 2.0, "q_factor": 0.8},
        ],
        "compressor": {
            "threshold_db": -18,
            "ratio": 3.5,
            "attack_ms": 8,
            "release_ms": 120,
            "makeup_gain_db": 2.5,
        },
        "reverb": {
            "decay_s": 2.5,
            "pre_delay_ms": 25,
            "mix": 0.25,
            "damping": 0.55,
            "room_size": 0.6,
        }
    }

    preset_manager.save_preset("my_custom_preset", custom_config)

    # Initialize synthesizer with custom preset
    glados = GLaDOSRuSynthesizer(preset_name="my_custom_preset")

    # Generate speech
    text = "Это пример с пользовательскими настройками эквалайзера и ревербератора."
    audio = glados.generate_speech_audio(text)

    # Save to file
    output_path = Path("output_custom.wav")
    sf.write(output_path, audio, glados.sample_rate)
    logger.info(f"Audio saved to {output_path}")


def realtime_parameter_adjustment():
    """Example 3: Real-time parameter adjustment."""
    logger.info("=== Example 3: Real-time Adjustment ===")

    # Initialize synthesizer
    glados = GLaDOSRuSynthesizer()

    # Generate base audio
    text = "Тестирование параметров обработки аудио."
    logger.info(f"Generating with default settings...")
    audio1 = glados.generate_speech_audio(text)
    sf.write("output_default.wav", audio1, glados.sample_rate)

    # Adjust EQ band
    logger.info("Adjusting EQ band 2 (3200 Hz peak)...")
    glados.update_eq_band(2, gain_db=8.0)  # Increase presence boost
    audio2 = glados.generate_speech_audio(text)
    sf.write("output_eq_boosted.wav", audio2, glados.sample_rate)

    # Adjust compressor
    logger.info("Adjusting compressor...")
    glados.update_compressor(ratio=6.0, threshold_db=-15)
    audio3 = glados.generate_speech_audio(text)
    sf.write("output_compressed.wav", audio3, glados.sample_rate)

    # Adjust reverb
    logger.info("Adjusting reverb...")
    glados.update_reverb(mix=0.5, decay_s=4.5)
    audio4 = glados.generate_speech_audio(text)
    sf.write("output_reverb_heavy.wav", audio4, glados.sample_rate)

    # Save current settings as preset
    glados.save_preset("my_adjusted_preset")
    logger.info("Settings saved as 'my_adjusted_preset'")


def batch_processing():
    """Example 4: Batch processing multiple texts."""
    logger.info("=== Example 4: Batch Processing ===")

    glados = GLaDOSRuSynthesizer()

    texts = [
        "Добро пожаловать в центр обогащения Aperture Science.",
        "Пожалуйста, обратите внимание, что мы добавили торт.",
        "Торт не является ложью.",
        "Все испытания будут проходить под наблюдением.",
    ]

    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)

    for i, text in enumerate(texts, 1):
        logger.info(f"Processing {i}/{len(texts)}: {text}")
        audio = glados.generate_speech_audio(text)

        output_path = output_dir / f"line_{i:02d}.wav"
        sf.write(output_path, audio, glados.sample_rate)

    logger.info(f"Batch processing complete. Files saved to {output_dir}")


def without_rvc():
    """Example 5: Using only Silero TTS without RVC."""
    logger.info("=== Example 5: Without RVC ===")

    # Initialize with RVC disabled
    glados = GLaDOSRuSynthesizer(enable_rvc=False)

    text = "Это пример без конверсии голоса RVC."
    audio = glados.generate_speech_audio(text)

    output_path = Path("output_no_rvc.wav")
    sf.write(output_path, audio, glados.sample_rate)
    logger.info(f"Audio saved to {output_path}")


def standalone_audio_processing():
    """Example 6: Using audio processing pipeline standalone."""
    logger.info("=== Example 6: Standalone Audio Processing ===")

    # Load existing audio
    audio, sr = sf.read("output_basic.wav")

    # Initialize audio processor
    processor = AudioProcessingPipeline(sample_rate=sr)
    processor.load_glados_preset()

    # Process audio
    processed_audio = processor.process(audio)

    # Save processed audio
    output_path = Path("output_reprocessed.wav")
    sf.write(output_path, processed_audio, sr)
    logger.info(f"Reprocessed audio saved to {output_path}")


def interactive_tui():
    """Example 7: Interactive TUI for parameter adjustment."""
    logger.info("=== Example 7: Interactive TUI ===")

    # Initialize pipeline
    pipeline = AudioProcessingPipeline()
    pipeline.load_glados_preset()

    # Run TUI
    logger.info("Launching interactive TUI...")
    logger.info("Controls:")
    logger.info("  q - Quit")
    logger.info("  s - Save preset")
    logger.info("  l - Load preset")
    logger.info("  r - Reset to default")

    run_audio_processor_tui(pipeline)


def main():
    """Run all examples."""
    print("GLaDOS Russian TTS Examples")
    print("=" * 50)
    print()
    print("Available examples:")
    print("1. Basic usage")
    print("2. Custom preset")
    print("3. Real-time parameter adjustment")
    print("4. Batch processing")
    print("5. Without RVC")
    print("6. Standalone audio processing")
    print("7. Interactive TUI")
    print("0. Run all examples")
    print()

    choice = input("Enter example number (or 'q' to quit): ").strip()

    if choice == 'q':
        return

    examples = {
        '0': lambda: [fn() for fn in [
            basic_usage,
            with_custom_preset,
            realtime_parameter_adjustment,
            batch_processing,
            without_rvc,
            standalone_audio_processing,
        ]],
        '1': basic_usage,
        '2': with_custom_preset,
        '3': realtime_parameter_adjustment,
        '4': batch_processing,
        '5': without_rvc,
        '6': standalone_audio_processing,
        '7': interactive_tui,
    }

    example_fn = examples.get(choice)
    if example_fn:
        try:
            example_fn()
        except Exception as e:
            logger.error(f"Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
