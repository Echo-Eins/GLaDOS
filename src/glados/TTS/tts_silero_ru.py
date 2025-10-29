"""Silero TTS V4 Russian implementation for GLaDOS.

This module provides Russian text-to-speech synthesis using Silero V4 models
with the 'xenia' speaker voice.
"""

import torch
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from pathlib import Path
from typing import Callable

class SileroRuSynthesizer:
    """Russian TTS synthesizer using Silero V4 models.

    This synthesizer uses the Silero TTS V4 model with the 'xenia' speaker
    for high-quality Russian speech synthesis.
    """

    def __init__(
        self,
        speaker: str = "xenia",
        sample_rate: int = 48000,
        device: str | None = None,
        use_fp16: bool = True,
    ):
        """Initialize Silero Russian TTS synthesizer.

        Args:
            speaker: Speaker name (default: 'xenia')
                Available: 'aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'
            sample_rate: Audio sample rate (8000, 24000, or 48000)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision on CUDA for faster inference (default: True)

        Note:
            V4 model has automatic stress placement (put_accent) built-in.
        """
        self.speaker = speaker
        self.sample_rate = sample_rate

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # FP16 only works on CUDA
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'

        logger.info(f"Initializing Silero Russian TTS on {self.device}")
        if self.use_fp16:
            logger.info("FP16 precision enabled for faster inference")

        try:
            # Load Silero V4 Russian model
            # Returns: (model, example_text)
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru'
            )

            # Move model to target device first
            if hasattr(model, "to") and callable(getattr(model, "to")):
                to_result = model.to(self.device)
                if to_result is not None:
                    model = to_result
            else:
                logger.debug(
                    "Silero model does not expose a .to() helper; skipping device move"
                )

            # Determine desired precision (FP16 only when CUDA is available)
            target_dtype = torch.float16 if self.use_fp16 else torch.float32

            if self.use_fp16:
                try:
                    model = self._ensure_precision(model, torch.float16, prefer_attr="half")

                    logger.info("Model converted to FP16 precision")

                except Exception as precision_error:
                    logger.warning(
                    "Failed to convert model to FP16: {}. Falling back to FP32 precision.",
                    precision_error,
                    )
                    self.use_fp16 = False
                    target_dtype = torch.float32

            if not self.use_fp16 and target_dtype != torch.float32:
                target_dtype = torch.float32

            if target_dtype == torch.float32:
                try:
                    model = self._ensure_precision(model, torch.float32, prefer_attr="float")
                except Exception as precision_error:
                    logger.warning(
                        "Failed to enforce FP32 precision on Silero model: {}",
                        precision_error,
                    )

            # Ensure the model is in evaluation mode
            if hasattr(model, "eval") and callable(getattr(model, "eval")):
                eval_result = model.eval()
                if eval_result is not None:
                    model = eval_result
            else:
                logger.debug("Silero model does not expose eval(); skipping eval mode switch")

            # Save reference to model
            self.model = model

            logger.success(
                f"Silero V4 Russian TTS loaded successfully with speaker '{speaker}' "
                f"on {self.device} (FP16: {self.use_fp16})"
            )

        except Exception as e:
            logger.error(f"Failed to load Silero TTS model: {e}")
            raise

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Generate speech audio from Russian text.

        Args:
            text: Russian text to synthesize

        Returns:
            Audio samples as float32 numpy array
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return np.array([], dtype=np.float32)

        try:
            # Generate audio using Silero V4
            # Note: V4 has automatic stress (put_accent) built-in
            with torch.no_grad():
                # Silero models handle FP16 internally if model was converted
                # Don't use autocast - it can cause issues with custom models
                audio_tensor = self.model.apply_tts(
                    text=text,
                    speaker=self.speaker,
                    sample_rate=self.sample_rate
                )

            # Convert to numpy array
            if isinstance(audio_tensor, torch.Tensor):
                audio = audio_tensor.cpu().float().numpy()  # Convert to FP32 for CPU
            else:
                audio = np.array(audio_tensor, dtype=np.float32)

            # Ensure it's 1D
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Ensure float32
            audio = audio.astype(np.float32)

            logger.debug(f"Generated {len(audio)/self.sample_rate:.2f}s of audio for text: '{text[:50]}...'")

            # CRITICAL: Clear CUDA cache to prevent memory/cache accumulation
            # After 15+ TTS generations, accumulated cache can cause audio distortion
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after TTS generation")

            return audio

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            logger.exception(e)
            return np.array([], dtype=np.float32)

    def save_audio(self, audio: NDArray[np.float32], path: Path | str) -> None:
        """Save audio to WAV file.

        Args:
            audio: Audio samples
            path: Output file path
        """
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(path, audio, self.sample_rate)
        logger.info(f"Audio saved to {path}")

    def _ensure_precision(
            self,
            model: torch.nn.Module,
            dtype: torch.dtype,
            *,
            prefer_attr: str,
    ) -> torch.nn.Module:
        """Ensure that the Silero model tensors use the requested dtype."""

        attr: Callable[[], torch.nn.Module] | None = getattr(model, prefer_attr, None)
        if callable(attr):
            result = attr()
            if result is not None:
                model = result
        else:
            logger.debug(
                "Silero model does not expose %s(); attempting manual dtype conversion",
                prefer_attr,
            )

        # Check if the model has parameters() and buffers() methods
        # Silero models may not expose these directly
        if not (hasattr(model, 'parameters') and callable(getattr(model, 'parameters'))):
            logger.debug(
                "Model does not expose parameters() method. "
                "Precision conversion via %s() may have succeeded, or model uses custom structure.",
                prefer_attr,
            )
            return model

        if not (hasattr(model, 'buffers') and callable(getattr(model, 'buffers'))):
            logger.debug(
                "Model does not expose buffers() method. "
                "Skipping buffer conversion."
            )
            # Still try to convert parameters
            try:
                for param in model.parameters(recurse=True):
                    param.data = param.data.to(dtype=dtype, device=param.device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(dtype=dtype, device=param.grad.device)
            except Exception as e:
                logger.debug(f"Could not convert parameters: {e}")
            return model

        # Standard PyTorch module - convert parameters and buffers
        try:
            for param in model.parameters(recurse=True):
                param.data = param.data.to(dtype=dtype, device=param.device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(dtype=dtype, device=param.grad.device)

            for buffer in model.buffers(recurse=True):
                buffer.data = buffer.data.to(dtype=dtype, device=buffer.device)
        except Exception as e:
            logger.debug(f"Could not convert model precision manually: {e}")

        return model

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
