from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from loguru import logger
import soundfile as sf

def _wrap_ffmpeg_error(exc: FileNotFoundError) -> RuntimeError:
    """Create a user-friendly error when the ffmpeg binary is missing."""

    missing_binary = exc.filename or "ffmpeg"
    message = (
        "GigaAM ASR requires the external 'ffmpeg' executable to be installed and "
        "available on the system PATH. "
        f"Executable '{missing_binary}' could not be found. "
        "Install ffmpeg and restart GLaDOS."
    )

    return RuntimeError(message)


class AudioTranscriber:
    """GigaAM-based automatic speech recognition with emotion analysis."""

    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        _model_path: str | None = None,
        *,
        language: str = "ru",
        device: str | None = None,
        fp16_encoder: bool = True,
    ) -> None:
        try:
            import torch
            import gigaam
            from glados.utils.gigaam_patches import apply_gigaam_patches
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "GigaAM ASR requires the optional 'gigaam' and 'torch' dependencies. "
                "Install them with 'pip install glados[ru]' or add them manually."
            ) from exc

        # Apply patches to suppress warnings
        apply_gigaam_patches()

        self.language = language.lower()
        if self.language not in {"ru", "en"}:
            raise ValueError(f"Unsupported language for GigaAM transcriber: {language}")

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device

        logger.info(f"GigaAM inference backend: {self.device.upper()}")

        self._asr_model = gigaam.load_model("rnnt", fp16_encoder=fp16_encoder, device=resolved_device)
        self._emotion_model = gigaam.load_model("emo", fp16_encoder=fp16_encoder, device=resolved_device)

    def transcribe(self, audio_source: NDArray[Any]) -> str:
        text, _ = self.transcribe_with_emotions(audio_source)
        return text

    def transcribe_file(self, audio_path: Path) -> str:
        try:
            return str(self._asr_model.transcribe(str(audio_path)))
        except FileNotFoundError as exc:  # pragma: no cover - depends on environment
            raise _wrap_ffmpeg_error(exc) from exc

    def transcribe_with_emotions(
        self, audio_source: NDArray[Any] | Path
    ) -> tuple[str, Mapping[str, float] | None]:
        if isinstance(audio_source, Path):
            return self._decode_with_emotions(audio_source)

        audio_array = np.asarray(audio_source, dtype=np.float32)
        if audio_array.ndim != 1:
            audio_array = audio_array.reshape(-1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)

        try:
            sf.write(temp_path, audio_array, self.SAMPLE_RATE)
            return self._decode_with_emotions(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _decode_with_emotions(self, audio_path: Path) -> tuple[str, Mapping[str, float]]:
        try:
            transcription = str(self._asr_model.transcribe(str(audio_path)))
            emotion_probs = self._emotion_model.get_probs(str(audio_path))
        except FileNotFoundError as exc:  # pragma: no cover - depends on environment
            raise _wrap_ffmpeg_error(exc) from exc
        except Exception as e:
            # If emotion model fails, return transcription with no emotions
            logger.error(f"Emotion model failed: {e}")
            transcription = str(self._asr_model.transcribe(str(audio_path)))
            return transcription, {}

        # Sanitize emotion values to handle NaN/inf cases
        formatted = {}
        has_nan = False
        for label, value in emotion_probs.items():
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                formatted[label] = 0.0
                has_nan = True
            else:
                formatted[label] = round(float_val, 4)

        if has_nan:
            logger.warning(
                f"GigaAM emotion model returned NaN/inf values. "
                f"Original: {emotion_probs}, Sanitized: {formatted}"
            )

        return transcription, formatted