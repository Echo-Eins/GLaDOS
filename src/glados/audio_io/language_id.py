"""Language Identification (LID) module using Silero Language Classifier.

This module provides language detection capabilities to route audio segments
to appropriate language-specific processing pipelines (RU/EN branches).
"""

from pathlib import Path
import tempfile
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import soundfile as sf
import torch
from loguru import logger


LanguageCode = Literal["ru", "en", "de", "es"]


class SileroLanguageID:
    """Silero-based Language Identification for audio segments.

    Uses Silero Language Classifier (4-language or 95-language model)
    to detect the language of speech in audio segments.

    Supported languages (4L model): ru, en, de, es
    """

    SAMPLE_RATE: int = 16000
    MIN_CONFIDENCE: float = 0.7  # Default confidence threshold

    def __init__(
        self,
        model_type: Literal["4lang", "95lang"] = "4lang",
        device: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize Silero Language ID model.

        Args:
            model_type: Model variant ('4lang' for ru/en/de/es, '95lang' for more)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence for language detection
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Silero Language ID ({model_type}) on {self.device}")

        try:
            # Load Silero Language Classifier
            # For 4-language model (ru, en, de, es)
            if model_type == "4lang":
                self.model, self.languages = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_lang_detector'
                )
            else:
                # 95-language model
                self.model, self.languages = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_lang_detector_95lang'
                )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.success(
                f"Silero Language ID loaded successfully. "
                f"Supported languages: {self.languages}"
            )

        except Exception as e:
            logger.error(f"Failed to load Silero Language ID: {e}")
            raise

    def detect(
        self,
        audio: NDArray[np.float32]
    ) -> tuple[str, float]:
        """Detect language from audio segment.

        Args:
            audio: Audio samples (16kHz mono float32)

        Returns:
            Tuple of (language_code, confidence)
            Example: ("ru", 0.95)
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided to LID")
            return ("unknown", 0.0)

        # Normalize audio
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # Convert to mono

        # Ensure proper amplitude range [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        try:
            # Silero LID expects tensor input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)

            try:
                # Write to temp file (Silero LID expects file input)
                sf.write(temp_path, audio, self.SAMPLE_RATE)

                # Run detection
                with torch.no_grad():
                    lang_probs = self.model(str(temp_path))

                # Get top prediction
                if isinstance(lang_probs, dict):
                    # Model returns dict of {lang: prob}
                    sorted_langs = sorted(
                        lang_probs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    lang_code, confidence = sorted_langs[0]
                else:
                    # Model returns tensor
                    confidence, lang_idx = torch.max(lang_probs, dim=-1)
                    confidence = confidence.item()
                    lang_code = self.languages[lang_idx.item()]

                # Convert to float if needed
                if isinstance(confidence, torch.Tensor):
                    confidence = confidence.item()

                logger.debug(
                    f"Language detection: {lang_code} (confidence: {confidence:.3f})"
                )

                return (lang_code, float(confidence))

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            logger.exception(e)
            return ("unknown", 0.0)

    def detect_with_threshold(
        self,
        audio: NDArray[np.float32],
        threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """Detect language with confidence threshold.

        Args:
            audio: Audio samples (16kHz mono float32)
            threshold: Confidence threshold (uses instance default if None)

        Returns:
            Tuple of (language_code or None, confidence)
            Returns None for language if confidence < threshold
        """
        threshold = threshold or self.confidence_threshold
        lang_code, confidence = self.detect(audio)

        if confidence < threshold:
            logger.warning(
                f"Language detection confidence {confidence:.3f} below "
                f"threshold {threshold:.3f}, treating as uncertain"
            )
            return (None, confidence)

        return (lang_code, confidence)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
