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

        use_fp16 = fp16_encoder
        if resolved_device == "cpu" and fp16_encoder:
            logger.debug(
                "Disabling fp16 encoder precision for CPU inference to avoid numerical issues."
            )
            use_fp16 = False

        logger.info(f"GigaAM inference backend: {self.device.upper()}")
        logger.info(f"GigaAM encoder precision: {'fp16' if use_fp16 else 'fp32'}")

        #self._asr_model = gigaam.load_model("rnnt", fp16_encoder=fp16_encoder, device=resolved_device)
        #self._emotion_model = gigaam.load_model("emo", fp16_encoder=fp16_encoder, device=resolved_device)

        # Always use the resolved precision when loading the models.  Passing the
        # original ``fp16_encoder`` flag here would force fp16 even on CPU and
        # quickly lead to NaNs from the Torch runtime after the first
        # transcription round.
        self._asr_model = gigaam.load_model("rnnt", fp16_encoder=use_fp16, device=resolved_device)
        self._emotion_model = gigaam.load_model("emo", fp16_encoder=use_fp16, device=resolved_device)

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

        # Diagnostic logging
        duration_s = len(audio_array) / self.SAMPLE_RATE
        rms = np.sqrt(np.mean(audio_array**2))
        max_val = np.max(np.abs(audio_array))
        logger.debug(
            f"GigaAM input audio: duration={duration_s:.3f}s, samples={len(audio_array)}, "
            f"RMS={rms:.6f}, max_abs={max_val:.6f}, dtype={audio_array.dtype}"
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)

        try:
            sf.write(temp_path, audio_array, self.SAMPLE_RATE)
            file_size = temp_path.stat().st_size
            logger.debug(f"GigaAM temp WAV file: {temp_path} (size={file_size} bytes)")
            return self._decode_with_emotions(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _decode_with_emotions(self, audio_path: Path) -> tuple[str, Mapping[str, float]]:
        try:
            # Log before transcription
            logger.debug(f"GigaAM calling ASR model on: {audio_path}")
            transcription = str(self._asr_model.transcribe(str(audio_path)))
            logger.debug(f"GigaAM ASR raw result: '{transcription}' (length={len(transcription)})")

            # Log before emotion analysis
            logger.debug(f"GigaAM calling Emotion model on: {audio_path}")
            emotion_probs = self._emotion_model.get_probs(str(audio_path))
            logger.debug(f"GigaAM Emotion raw result: {dict(emotion_probs)}")

            # CRITICAL: Clear CUDA cache to prevent memory accumulation
            # GigaAM models don't have explicit reset(), so we clear torch cache
            if self.device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after GigaAM processing")

        except FileNotFoundError as exc:  # pragma: no cover - depends on environment
            raise _wrap_ffmpeg_error(exc) from exc
        except Exception as e:
            # If emotion model fails, return transcription with no emotions
            logger.error(f"Emotion model failed with exception: {type(e).__name__}: {e}")
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