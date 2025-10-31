"""Whisper ASR implementation for English speech recognition.

This module provides English automatic speech recognition using OpenAI's Whisper model,
specifically optimized for the EN-Branch pipeline.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from loguru import logger
import soundfile as sf


class WhisperTranscriber:
    """Whisper-based English ASR for EN-Branch pipeline.

    Uses openai/whisper-small.en or local equivalent for fast,
    accurate English speech recognition.
    """

    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        model_name: str = "small.en",
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        """Initialize Whisper ASR model.

        Args:
            model_name: Whisper model variant ('tiny.en', 'base.en', 'small.en', 'medium.en')
            device: Device to run on ('cpu', 'cuda', or None for auto)
            fp16: Use FP16 precision on CUDA (default: True)
        """
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "Whisper ASR requires 'openai-whisper' package. "
                "Install with: pip install openai-whisper"
            ) from exc

        self.model_name = model_name

        # Determine device
        if device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        self.device = resolved_device

        # FP16 only works on CUDA
        self.fp16 = fp16 and resolved_device == "cuda"

        logger.info(f"Whisper inference backend: {self.device.upper()}")
        logger.info(f"Whisper precision: {'FP16' if self.fp16 else 'FP32'}")

        try:
            # Load Whisper model
            logger.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(
                name=model_name,
                device=resolved_device,
            )

            # Set to eval mode
            self.model.eval()

            logger.success(
                f"Whisper-{model_name} loaded successfully on {resolved_device} "
                f"(FP16: {self.fp16})"
            )

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, audio_source: NDArray[Any]) -> str:
        """Transcribe audio to text.

        Args:
            audio_source: Audio samples (16kHz mono float32)

        Returns:
            Transcribed text string
        """
        audio_array = np.asarray(audio_source, dtype=np.float32)
        if audio_array.ndim != 1:
            audio_array = audio_array.reshape(-1)

        duration_s = len(audio_array) / self.SAMPLE_RATE
        rms = np.sqrt(np.mean(audio_array**2))
        max_val = np.max(np.abs(audio_array))

        logger.debug(
            f"Whisper input: {duration_s:.2f}s, {len(audio_array)} samples, "
            f"RMS={rms:.4f}, max={max_val:.4f}"
        )

        # Check for silence
        if rms < 0.001 or max_val < 0.001:
            logger.warning("Whisper received near-silent audio")
            return ""

        try:
            # Whisper expects audio normalized to [-1, 1]
            if max_val > 0:
                audio_array = audio_array / max_val

            # Transcribe using Whisper
            # language="en" for English-only, task="transcribe"
            result = self.model.transcribe(
                audio=audio_array,
                language="en",
                task="transcribe",
                fp16=self.fp16,
                verbose=False,
            )

            transcription = result.get("text", "").strip()

            logger.debug(
                f"Whisper transcription: '{transcription}' "
                f"(length={len(transcription)})"
            )

            # CRITICAL: Clear CUDA cache to prevent memory accumulation
            if self.device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after Whisper processing")

            return transcription

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            logger.exception(e)
            return ""

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text string
        """
        try:
            # Load audio file
            audio, sr = sf.read(audio_path)

            # Resample if needed (Whisper expects 16kHz)
            if sr != self.SAMPLE_RATE:
                logger.warning(
                    f"Audio sample rate {sr} != {self.SAMPLE_RATE}, "
                    f"resampling may be needed"
                )

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            return self.transcribe(audio)

        except Exception as e:
            logger.error(f"Failed to transcribe file {audio_path}: {e}")
            return ""

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
