"""Language Identification (LID) module using SpeechBrain VoxLingua107.

This module provides language detection capabilities to route audio segments
to appropriate language-specific processing pipelines (RU/EN branches).

Uses speechbrain/lang-id-voxlingua107-ecapa from HuggingFace.
Supports 107 languages but configured specifically for Russian/English detection.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch
from loguru import logger


LanguageCode = Literal["ru", "en", "unknown"]


class SpeechBrainLanguageID:
    """SpeechBrain VoxLingua107 Language Identification for audio segments.

    Uses ECAPA-TDNN model trained on 107 languages.
    Configured for Russian/English detection with warnings for other languages.

    Achieves 6.7% error rate on VoxLingua107 dev set.
    """

    SAMPLE_RATE: int = 16000
    SUPPORTED_LANGUAGES = {"ru", "en"}  # Only Russian and English supported

    def __init__(
        self,
        model_name: str = "speechbrain/lang-id-voxlingua107-ecapa",
        device: str | None = None,
        confidence_threshold: float = 0.7,
        default_language: Literal["ru", "en"] = "ru",
    ):
        """Initialize SpeechBrain Language ID model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence for language detection
            default_language: Default language when confidence < threshold or unsupported
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.default_language = default_language

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing SpeechBrain Language ID on {self.device}")
        logger.info(f"Supported languages: {self.SUPPORTED_LANGUAGES} (others will trigger warnings)")

        try:
            from speechbrain.inference.classifiers import EncoderClassifier

            # Load model from HuggingFace
            cache_dir = Path.home() / '.cache' / 'speechbrain'
            cache_dir.mkdir(parents=True, exist_ok=True)

            self.classifier = EncoderClassifier.from_hparams(
                source=model_name,
                savedir=str(cache_dir / "lang-id-voxlingua107-ecapa"),
                run_opts={"device": str(self.device)},
            )

            # Configure label encoder to the expected number of categories
            label_encoder = getattr(self.classifier.hparams, "label_encoder", None)
            if label_encoder is not None and hasattr(label_encoder, "expect_len"):
                try:
                    label_encoder.expect_len(107)
                except Exception as enc_exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Unable to set label encoder expected length: %s", enc_exc
                    )

            # Language code mapping (VoxLingua107 uses ISO 639-3 codes)
            # Map from VoxLingua codes to our 2-letter codes
            self.lang_code_map = {
                "rus": "ru",  # Russian
                "eng": "en",  # English
            }

            logger.success(
                f"SpeechBrain Language ID loaded successfully "
                f"(VoxLingua107 ECAPA-TDNN, {len(self.SUPPORTED_LANGUAGES)} languages supported)"
            )

        except Exception as e:
            logger.error(f"Failed to load SpeechBrain Language ID: {e}")
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
            - language_code: "ru", "en", or "unknown"
            - confidence: float in [0, 1]

        Note:
            If detected language is not in SUPPORTED_LANGUAGES (ru/en),
            a warning is logged and default_language is returned with low confidence.
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
            # Convert to torch tensor (SpeechBrain expects [batch, time])
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

            # Run classification
            with torch.no_grad():
                prediction = self.classifier.classify_batch(audio_tensor)

            # Extract outputs from SpeechBrain tuple: (posteriors, scores, index, text_lab)
            _, scores, indices, labels = prediction

            # Derive textual label
            if isinstance(labels, (list, tuple)):
                text_lab = labels[0]
            elif torch.is_tensor(labels):
                text_lab = labels.squeeze()[0]
            else:
                text_lab = str(labels)

            if isinstance(text_lab, torch.Tensor):
                text_lab = text_lab.item() if text_lab.ndim == 0 else text_lab.tolist()
            if isinstance(text_lab, list):
                text_lab = text_lab[0]
            text_lab = str(text_lab)

            # Prepare score tensor for confidence computation
            if torch.is_tensor(scores):
                score_tensor = scores.squeeze()
            else:
                score_tensor = torch.as_tensor(scores)

            if score_tensor.ndim == 0:
                score_tensor = score_tensor.unsqueeze(0)

            if score_tensor.numel() == 0:
                logger.error(
                    "Language ID model returned an empty score tensor; falling back to default language"
                )
                return (self.default_language, 0.0)

            # Predicted index may be tensor/list
            if torch.is_tensor(indices):
                pred_idx = int(indices.squeeze().item())
            elif isinstance(indices, (list, tuple)):
                pred_idx = int(indices[0])
            else:
                pred_idx = int(indices)

            if pred_idx < 0 or pred_idx >= score_tensor.numel():
                logger.error(
                    "Language ID predicted index %s outside score tensor bounds %s; using max score instead",
                    pred_idx,
                    score_tensor.shape,
                )
                pred_idx = int(torch.argmax(score_tensor).item())

            confidence = torch.softmax(score_tensor, dim=0)[pred_idx].item()

            # Map VoxLingua code to our 2-letter code
            lang_code = self.lang_code_map.get(
                text_lab, text_lab[:2] if len(text_lab) >= 2 else text_lab
            )

            # Check if language is supported
            if lang_code not in self.SUPPORTED_LANGUAGES:
                logger.warning(
                    f"⚠️  Detected unsupported language: {text_lab} ({lang_code}) "
                    f"with confidence {confidence:.3f}. "
                    f"Only {self.SUPPORTED_LANGUAGES} are supported. "
                    f"Routing to default: {self.default_language}"
                )
                return (self.default_language, 0.5)  # Medium confidence for fallback

            logger.debug(
                f"Language detection: {text_lab} → {lang_code} "
                f"(confidence: {confidence:.3f})"
            )

            return (lang_code, float(confidence))

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
                f"threshold {threshold:.3f}, using default language: {self.default_language}"
            )
            return (self.default_language, confidence)

        return (lang_code, confidence)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'classifier'):
            del self.classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Backward compatibility alias
SileroLanguageID = SpeechBrainLanguageID
