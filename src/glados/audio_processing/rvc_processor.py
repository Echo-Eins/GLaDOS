"""RVC (Retrieval-based Voice Conversion) processor for GLaDOS voice.

This module handles voice conversion using pre-trained RVC models
to transform input speech into GLaDOS-like voice.
"""

import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from loguru import logger
import tempfile
import os

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available. RVC processing will be limited.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. RVC index search will be disabled.")

try:
    from rvc_python.infer import RVCInference as RVCPythonInference
    RVC_PYTHON_AVAILABLE = True
    logger.info("rvc-python library detected")
except ImportError:
    RVC_PYTHON_AVAILABLE = False
    logger.warning("rvc-python not available. RVC will use fallback mode.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


class RVCProcessor:
    """RVC voice conversion processor for GLaDOS voice.

    Converts input audio to GLaDOS voice using pre-trained RVC model.
    Uses rvc-python library if available, otherwise falls back to basic processing.
    """

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        device: str | None = None,
        f0_method: str = "harvest",
        f0_up_key: int = 0,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        """Initialize RVC processor.

        Args:
            model_path: Path to RVC model (.pth file)
            index_path: Path to feature index (.index file)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            f0_method: F0 extraction method ('harvest', 'pm', 'dio', 'crepe', 'rmvpe')
            f0_up_key: Pitch shift in semitones
            index_rate: Feature index influence (0.0 to 1.0)
            filter_radius: Median filter radius for F0 smoothing (0-7)
            rms_mix_rate: RMS envelope mixing rate (0.0 to 1.0)
            protect: Consonant protection (0.0 to 0.5)
        """
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.f0_method = f0_method
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect

        # Determine device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing RVC processor on {self.device}")

        # Check model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        # Initialize RVC inference
        self.rvc = None
        if RVC_PYTHON_AVAILABLE:
            self._init_rvc_python()
        else:
            logger.warning("RVC will run in fallback mode (no voice conversion)")

        logger.success("RVC processor initialized")

    def _init_rvc_python(self) -> None:
        """Initialize rvc-python inference engine."""
        try:
            logger.info("Initializing rvc-python inference...")
            self.rvc = RVCPythonInference(device=self.device)

            # Load model with index
            logger.info(f"Loading RVC model: {self.model_path}")
            model_version = "v2"  # May need to detect from model
            index_path_str = str(self.index_path) if self.index_path and self.index_path.exists() else ""

            if index_path_str:
                logger.info(f"Using feature index: {self.index_path}")
            else:
                logger.warning("No feature index provided")

            self.rvc.load_model(
                model_path_or_name=str(self.model_path),
                version=model_version,
                index_path=index_path_str
            )

            # Set parameters
            self.rvc.set_params(
                f0up_key=self.f0_up_key,
                f0method=self.f0_method,
                index_rate=self.index_rate,
                filter_radius=self.filter_radius,
                rms_mix_rate=self.rms_mix_rate,
                protect=self.protect,
            )

            logger.success("rvc-python initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize rvc-python: {e}")
            logger.exception(e)
            self.rvc = None

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio through RVC voice conversion.

        Args:
            audio: Input audio samples (float32, -1.0 to 1.0)
            input_sample_rate: Sample rate of input audio

        Returns:
            Converted audio samples (float32)
        """
        if len(audio) == 0:
            return audio

        # If RVC not available, return original audio
        if self.rvc is None:
            logger.warning("RVC not available - returning original audio")
            return audio

        try:
            logger.debug(f"RVC processing: {len(audio)/input_sample_rate:.2f}s audio")

            # Use temporary files for RVC processing
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / "input.wav"
                output_path = Path(tmpdir) / "output.wav"

                # Write input audio
                if not SOUNDFILE_AVAILABLE:
                    logger.error("soundfile not available, cannot process audio")
                    return audio

                sf.write(input_path, audio, input_sample_rate)

                # Process with RVC
                logger.debug(f"Running RVC inference: {self.f0_method}, pitch={self.f0_up_key}")
                self.rvc.infer_file(
                    input_path=str(input_path),
                    output_path=str(output_path),
                )

                # Read result
                if output_path.exists():
                    converted_audio, sr = sf.read(output_path, dtype='float32')
                    logger.debug(f"RVC conversion complete: {len(converted_audio)/sr:.2f}s")

                    # Resample if needed
                    if sr != input_sample_rate and LIBROSA_AVAILABLE:
                        logger.debug(f"Resampling from {sr}Hz to {input_sample_rate}Hz")
                        converted_audio = librosa.resample(
                            converted_audio,
                            orig_sr=sr,
                            target_sr=input_sample_rate
                        )

                    return converted_audio.astype(np.float32)
                else:
                    logger.error("RVC output file not created")
                    return audio

        except Exception as e:
            logger.error(f"RVC processing failed: {e}")
            logger.exception(e)
            return audio  # Return original audio on error

    def update_params(
        self,
        f0_up_key: int | None = None,
        f0_method: str | None = None,
        index_rate: float | None = None,
        filter_radius: int | None = None,
        rms_mix_rate: float | None = None,
        protect: float | None = None,
    ) -> None:
        """Update RVC parameters.

        Args:
            f0_up_key: Pitch shift in semitones
            f0_method: F0 extraction method
            index_rate: Feature index influence
            filter_radius: Median filter radius
            rms_mix_rate: RMS envelope mixing rate
            protect: Consonant protection
        """
        if f0_up_key is not None:
            self.f0_up_key = f0_up_key
        if f0_method is not None:
            self.f0_method = f0_method
        if index_rate is not None:
            self.index_rate = index_rate
        if filter_radius is not None:
            self.filter_radius = filter_radius
        if rms_mix_rate is not None:
            self.rms_mix_rate = rms_mix_rate
        if protect is not None:
            self.protect = protect

        # Update RVC parameters if available
        if self.rvc is not None and hasattr(self.rvc, 'set_params'):
            self.rvc.set_params(
                f0up_key=self.f0_up_key,
                f0method=self.f0_method,
                index_rate=self.index_rate,
                filter_radius=self.filter_radius,
                rms_mix_rate=self.rms_mix_rate,
                protect=self.protect,
            )
            logger.debug("RVC parameters updated")


class SimpleRVCProcessor:
    """Simplified RVC processor with basic pitch shifting.

    This is a fallback implementation when full RVC is not available.
    """

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        device: str | None = None,
        pitch_shift: int = 0,
    ):
        """Initialize simple RVC processor.

        Args:
            model_path: Path to RVC model (not used in simple mode)
            index_path: Path to feature index (not used)
            device: Device to run on (not used)
            pitch_shift: Pitch shift in semitones
        """
        self.model_path = Path(model_path)
        self.pitch_shift = pitch_shift

        logger.warning(
            "SimpleRVCProcessor initialized - this only does basic pitch shifting, "
            "not real voice conversion. Install rvc-python for full RVC support."
        )

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio with basic pitch shifting.

        Args:
            audio: Input audio
            input_sample_rate: Input sample rate

        Returns:
            Pitch-shifted audio
        """
        if len(audio) == 0 or self.pitch_shift == 0:
            return audio

        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, cannot apply pitch shift")
            return audio

        try:
            import librosa
            logger.debug(f"Applying pitch shift: {self.pitch_shift} semitones")
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=input_sample_rate,
                n_steps=self.pitch_shift,
            )
            return shifted.astype(np.float32)
        except Exception as e:
            logger.error(f"Pitch shift failed: {e}")
            return audio


def create_rvc_processor(
    model_path: Path | str,
    index_path: Path | str | None = None,
    simple_mode: bool = False,
    **kwargs
) -> RVCProcessor | SimpleRVCProcessor:
    """Factory function to create appropriate RVC processor.

    Args:
        model_path: Path to RVC model
        index_path: Path to feature index
        simple_mode: Use simplified processor
        **kwargs: Additional arguments for processor

    Returns:
        RVC processor instance
    """
    if simple_mode or not RVC_PYTHON_AVAILABLE:
        if not simple_mode:
            logger.warning("rvc-python not available, using simple mode")
        return SimpleRVCProcessor(
            model_path,
            index_path,
            pitch_shift=kwargs.get('f0_up_key', 0),
            **{k: v for k, v in kwargs.items() if k in ['device']}
        )
    else:
        return RVCProcessor(model_path, index_path, **kwargs)
