"""RVC (Retrieval-based Voice Conversion) processor for GLaDOS voice.

This module handles voice conversion using pre-trained RVC models
to transform input speech into GLaDOS-like voice.
"""

import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from loguru import logger

try:
    # Try to import RVC dependencies
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. RVC index search will be disabled.")


class RVCProcessor:
    """RVC voice conversion processor for GLaDOS voice.

    Converts input audio to GLaDOS voice using pre-trained RVC model.
    """

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        device: str | None = None,
        f0_method: str = "harvest",
        f0_up_key: int = 0,
        index_rate: float = 0.75,
    ):
        """Initialize RVC processor.

        Args:
            model_path: Path to RVC model (.pth file)
            index_path: Path to feature index (.index file)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            f0_method: F0 extraction method ('harvest', 'pm', 'dio', 'crepe')
            f0_up_key: Pitch shift in semitones
            index_rate: Feature index influence (0.0 to 1.0)
        """
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.f0_method = f0_method
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing RVC processor on {self.device}")

        # Load model
        self._load_model()

        # Load index if available
        if self.index_path and self.index_path.exists() and FAISS_AVAILABLE:
            self._load_index()
        else:
            self.index = None
            if self.index_path:
                logger.warning(f"Index file not found or FAISS unavailable: {self.index_path}")

    def _load_model(self) -> None:
        """Load RVC model from checkpoint."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading RVC model from {self.model_path}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Store model configuration
            self.config = checkpoint.get('config', {})
            self.sample_rate = self.config.get('sample_rate', 48000)
            self.hop_size = self.config.get('hop_size', 512)
            self.f0 = self.config.get('if_f0', 1) == 1

            # Store weights
            self.weights = checkpoint.get('weight', checkpoint)

            logger.success(f"RVC model loaded successfully (sr={self.sample_rate})")

        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            raise

    def _load_index(self) -> None:
        """Load feature index for retrieval."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index loading")
            return

        try:
            logger.info(f"Loading feature index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            logger.success("Feature index loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load feature index: {e}")
            self.index = None

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio through RVC voice conversion.

        Args:
            audio: Input audio samples
            input_sample_rate: Sample rate of input audio

        Returns:
            Converted audio samples
        """
        if len(audio) == 0:
            return audio

        try:
            # Resample if needed
            if input_sample_rate != self.sample_rate:
                audio = self._resample(audio, input_sample_rate, self.sample_rate)

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)

            # Perform voice conversion
            with torch.no_grad():
                converted_audio = self._convert_voice(audio_tensor)

            # Convert back to numpy
            output = converted_audio.cpu().numpy()

            # Ensure float32
            output = output.astype(np.float32)

            logger.debug(f"RVC conversion complete: {len(output)/self.sample_rate:.2f}s")

            return output

        except Exception as e:
            logger.error(f"RVC processing failed: {e}")
            return audio  # Return original audio on error

    def _convert_voice(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Core voice conversion logic.

        This is a placeholder that would need actual RVC inference code.
        For now, it returns the input (passthrough).

        Args:
            audio_tensor: Input audio as torch tensor

        Returns:
            Converted audio as torch tensor
        """
        # TODO: Implement actual RVC inference
        # This would typically involve:
        # 1. Extract F0 (pitch)
        # 2. Extract features
        # 3. Query index for similar features
        # 4. Run through RVC model
        # 5. Synthesize output

        logger.warning("RVC conversion not fully implemented - using passthrough")
        return audio_tensor

    def _resample(
        self,
        audio: NDArray[np.float32],
        orig_sr: int,
        target_sr: int
    ) -> NDArray[np.float32]:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        import librosa

        if orig_sr == target_sr:
            return audio

        logger.debug(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32)


class SimpleRVCProcessor:
    """Simplified RVC processor using external inference tools.

    This version uses command-line RVC tools or simplified processing
    when full RVC implementation is not available.
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
            model_path: Path to RVC model
            index_path: Path to feature index
            device: Device to run on
            pitch_shift: Pitch shift in semitones
        """
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.pitch_shift = pitch_shift

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Simple RVC processor initialized with model: {self.model_path}")

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio (simplified version).

        Args:
            audio: Input audio
            input_sample_rate: Input sample rate

        Returns:
            Processed audio (currently passthrough)
        """
        logger.debug("Simple RVC processing (passthrough mode)")
        # This is a placeholder - actual implementation would use
        # external RVC tools or a simplified conversion algorithm
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
    if simple_mode:
        return SimpleRVCProcessor(model_path, index_path, **kwargs)
    else:
        return RVCProcessor(model_path, index_path, **kwargs)
