"""GLaDOS Russian voice synthesis pipeline.

Complete TTS pipeline: Silero TTS -> RVC -> Audio Processing (EQ/Compressor/Reverb)
"""

import time
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from loguru import logger

from .tts_silero_ru import SileroRuSynthesizer
from ..audio_processing.rvc_processor import create_rvc_processor
from ..audio_processing.audio_processor import AudioProcessingPipeline
from ..audio_processing.preset_manager import PresetManager
from ..utils.resources import resource_path


class GLaDOSRuSynthesizer:
    """Complete GLaDOS Russian voice synthesis pipeline.

    This synthesizer combines:
    1. Silero V4 Russian TTS (xenia speaker)
    2. RVC voice conversion to GLaDOS
    3. Audio processing (EQ, compression, reverb) for final polish
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        index_path: Path | str | None = None,
        preset_name: str = "glados_default",
        device: str | None = None,
        enable_rvc: bool = True,
        enable_audio_processing: bool = True,
    ):
        """Initialize GLaDOS Russian synthesizer.

        Args:
            model_path: Path to RVC model (default: models/TTS/GLaDOS_ru/ru_glados.pth)
            index_path: Path to RVC index (default: models/TTS/GLaDOS_ru/added_IVF424_Flat_nprobe_1_ru_glados_v2.index)
            preset_name: Audio processing preset name
            device: Device to run on ('cpu', 'cuda', or None for auto)
            enable_rvc: Enable RVC voice conversion
            enable_audio_processing: Enable audio post-processing
        """
        self.enable_rvc = enable_rvc
        self.enable_audio_processing = enable_audio_processing

        # Set default paths
        if model_path is None:
            model_path = resource_path("models/TTS/GLaDOS_ru/ru_glados.pth")
        if index_path is None:
            index_path = resource_path("models/TTS/GLaDOS_ru/added_IVF424_Flat_nprobe_1_ru_glados_v2.index")

        self.model_path = Path(model_path)
        self.index_path = Path(index_path)

        # Initialize Silero TTS with FP16 enabled on CUDA
        logger.info("Initializing Silero Russian TTS...")
        self.tts = SileroRuSynthesizer(
            speaker="xenia",
            sample_rate=48000,
            device=device,
            use_fp16=True,  # Enable FP16 for faster inference on CUDA
        )
        self.sample_rate = self.tts.sample_rate

        # Initialize RVC processor
        if self.enable_rvc:
            logger.info("Initializing RVC processor...")
            try:
                self.rvc = create_rvc_processor(
                    model_path=self.model_path,
                    index_path=self.index_path,
                    simple_mode=False,  # Use full RVC mode with inferrvc
                    device=device,
                    f0_method="rmvpe",  # F0 extraction method (rmvpe is best)
                    f0_up_key=0,  # Pitch shift (0 = no shift)
                    index_rate=0.75,  # Feature index influence
                    filter_radius=3,  # Median filter radius
                    protect=0.33,  # Consonant protection
                )
                logger.success("RVC processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RVC processor: {e}")
                logger.warning("RVC will be disabled")
                self.enable_rvc = False
                self.rvc = None
        else:
            self.rvc = None

        # Initialize audio processing pipeline
        if self.enable_audio_processing:
            logger.info("Initializing audio processing pipeline...")
            self.audio_processor = AudioProcessingPipeline(sample_rate=self.sample_rate)

            # Load preset
            presets_dir = resource_path("presets")
            self.preset_manager = PresetManager(presets_dir=presets_dir)
            try:
                preset_config = self.preset_manager.load_preset(preset_name)
                self.audio_processor.from_dict(preset_config)
                logger.success(f"Loaded preset: {preset_name}")
            except FileNotFoundError:
                logger.warning(f"Preset '{preset_name}' not found in {presets_dir}, using default GLaDOS preset")
                self.audio_processor.load_glados_preset()
            except Exception as e:
                logger.error(f"Failed to load preset: {e}")
                logger.warning("Using default GLaDOS preset")
                self.audio_processor.load_glados_preset()
        else:
            self.audio_processor = None
            self.preset_manager = None

        logger.success("GLaDOS Russian synthesizer initialized successfully")

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Generate GLaDOS Russian speech from text.

        Pipeline: Text -> Silero TTS -> RVC -> Audio Processing

        Args:
            text: Russian text to synthesize

        Returns:
            Processed audio samples
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return np.array([], dtype=np.float32)

        start_time = time.time()

        # Step 1: Generate base audio with Silero TTS
        logger.debug("Step 1: Silero TTS generation...")
        tts_start = time.time()
        audio = self.tts.generate_speech_audio(text)
        tts_time = time.time() - tts_start

        if len(audio) == 0:
            logger.warning("TTS generated empty audio")
            return audio

        logger.debug(f"TTS complete: {tts_time:.3f}s, audio length: {len(audio)/self.sample_rate:.2f}s")

        # Step 2: Apply RVC voice conversion
        if self.enable_rvc and self.rvc is not None:
            logger.debug("Step 2: RVC voice conversion...")
            rvc_start = time.time()
            audio = self.rvc.process(audio, input_sample_rate=self.sample_rate)
            rvc_time = time.time() - rvc_start
            logger.debug(f"RVC complete: {rvc_time:.3f}s")
        else:
            logger.debug("Step 2: RVC skipped (disabled)")

        # Step 3: Apply audio processing (EQ, compression, reverb)
        if self.enable_audio_processing and self.audio_processor is not None:
            logger.debug("Step 3: Audio processing (EQ/Comp/Reverb)...")
            proc_start = time.time()
            audio = self.audio_processor.process(audio)
            proc_time = time.time() - proc_start
            logger.debug(f"Audio processing complete: {proc_time:.3f}s")
        else:
            logger.debug("Step 3: Audio processing skipped (disabled)")

        total_time = time.time() - start_time
        audio_duration = len(audio) / self.sample_rate
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"GLaDOS RU synthesis complete: {total_time:.3f}s "
            f"({audio_duration:.2f}s audio, RTF={rtf:.2f})"
        )

        return audio

    def load_preset(self, preset_name: str) -> None:
        """Load an audio processing preset.

        Args:
            preset_name: Name of preset to load
        """
        if not self.enable_audio_processing or self.audio_processor is None:
            logger.warning("Audio processing is disabled, cannot load preset")
            return

        try:
            preset_config = self.preset_manager.load_preset(preset_name)
            self.audio_processor.from_dict(preset_config)
            logger.success(f"Loaded preset: {preset_name}")
        except Exception as e:
            logger.error(f"Failed to load preset '{preset_name}': {e}")

    def save_preset(self, preset_name: str) -> None:
        """Save current audio processing settings as a preset.

        Args:
            preset_name: Name for the preset
        """
        if not self.enable_audio_processing or self.audio_processor is None:
            logger.warning("Audio processing is disabled, cannot save preset")
            return

        try:
            config = self.audio_processor.to_dict()
            self.preset_manager.save_preset(preset_name, config)
            logger.success(f"Preset saved: {preset_name}")
        except Exception as e:
            logger.error(f"Failed to save preset '{preset_name}': {e}")

    def update_eq_band(self, band_index: int, **kwargs) -> None:
        """Update a specific EQ band.

        Args:
            band_index: Index of the band to update
            **kwargs: Parameters to update (freq, gain_db, q, filter_type)
        """
        if not self.enable_audio_processing or self.audio_processor is None:
            logger.warning("Audio processing is disabled")
            return

        if band_index >= len(self.audio_processor.eq.bands):
            logger.error(f"Invalid band index: {band_index}")
            return

        band = self.audio_processor.eq.bands[band_index]

        if 'freq' in kwargs:
            band.frequency = kwargs['freq']
        if 'gain_db' in kwargs:
            band.gain_db = kwargs['gain_db']
        if 'q' in kwargs:
            band.q_factor = kwargs['q']
        if 'filter_type' in kwargs:
            band.filter_type = kwargs['filter_type']

        # Rebuild filters
        self.audio_processor.eq._update_filters()
        logger.debug(f"Updated EQ band {band_index}")

    def update_compressor(self, **kwargs) -> None:
        """Update compressor settings.

        Args:
            **kwargs: Parameters to update (threshold_db, ratio, attack_ms, release_ms, makeup_gain_db)
        """
        if not self.enable_audio_processing or self.audio_processor is None:
            logger.warning("Audio processing is disabled")
            return

        comp = self.audio_processor.compressor

        if 'threshold_db' in kwargs:
            comp.threshold_db = kwargs['threshold_db']
        if 'ratio' in kwargs:
            comp.ratio = kwargs['ratio']
        if 'attack_ms' in kwargs:
            comp.attack_ms = kwargs['attack_ms']
            comp.attack_coef = np.exp(-1000.0 / (comp.attack_ms * comp.sample_rate))
        if 'release_ms' in kwargs:
            comp.release_ms = kwargs['release_ms']
            comp.release_coef = np.exp(-1000.0 / (comp.release_ms * comp.sample_rate))
        if 'makeup_gain_db' in kwargs:
            comp.makeup_gain_db = kwargs['makeup_gain_db']

        logger.debug("Updated compressor settings")

    def update_reverb(self, **kwargs) -> None:
        """Update reverb settings.

        Args:
            **kwargs: Parameters to update (decay_s, pre_delay_ms, mix, damping, room_size)
        """
        if not self.enable_audio_processing or self.audio_processor is None:
            logger.warning("Audio processing is disabled")
            return

        # Recreate reverb with new settings
        current = self.audio_processor.reverb.to_dict()
        current.update(kwargs)
        self.audio_processor.reverb.from_dict(current)

        logger.debug("Updated reverb settings")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'tts'):
            del self.tts
        if hasattr(self, 'rvc') and self.rvc is not None:
            del self.rvc
