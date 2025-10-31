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

    Uses Silero Language Classifier (95-language model)
    to detect the language of speech in audio segments.

    Supported languages: ru, en, de, es, and 91 more
    """

    SAMPLE_RATE: int = 16000
    MIN_CONFIDENCE: float = 0.7  # Default confidence threshold

    def __init__(
        self,
        model_type: Literal["4lang", "95lang"] = "95lang",
        device: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize Silero Language ID model.

        Args:
            model_type: Model variant (only '95lang' is supported, '4lang' will use 95lang)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence for language detection
        """
        self.model_type = "95lang"  # Only 95lang available
        self.confidence_threshold = confidence_threshold

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Silero Language ID (95lang) on {self.device}")

        try:
            # Load Silero Language Classifier from silero-vad repo
            # Returns: (model, lang_dict, lang_group_dict, utils)
            model, lang_dict, lang_group_dict, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_lang_detector_95',
                force_reload=False,
                onnx=False,
            )

            # Extract utils
            self.get_language_and_group, self.read_audio = utils

            # Move model to device
            self.model = model.to(self.device)
            self.model.eval()

            # Store language dictionaries
            self.lang_dict = lang_dict
            self.lang_group_dict = lang_group_dict

            # Map language names to codes for common languages
            # Silero returns full names like "Russian", "English"
            self.lang_name_to_code = {
                "Russian": "ru",
                "English": "en",
                "German": "de",
                "Spanish": "es",
                "French": "fr",
                "Italian": "it",
                "Portuguese": "pt",
                "Polish": "pl",
                "Ukrainian": "uk",
            }

            logger.success(
                f"Silero Language ID loaded successfully. "
                f"Supports 95 languages"
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
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio).to(self.device)

            # Run detection using Silero LID API
            with torch.no_grad():
                languages, language_groups = self.get_language_and_group(
                    audio_tensor,
                    self.model,
                    self.lang_dict,
                    self.lang_group_dict,
                    top_n=1  # Get top 1 prediction
                )

            # Extract top language
            if languages and len(languages) > 0:
                lang_info = languages[0]  # Top prediction
                lang_name = lang_info[0]  # Language name (e.g., "Russian")
                confidence = lang_info[1]  # Confidence score

                # Convert language name to code
                lang_code = self.lang_name_to_code.get(lang_name, lang_name.lower()[:2])

                logger.debug(
                    f"Language detection: {lang_name} ({lang_code}) "
                    f"(confidence: {confidence:.3f})"
                )

                return (lang_code, float(confidence))
            else:
                logger.warning("No language detected from audio")
                return ("unknown", 0.0)

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
