"""Language-based audio routing for parallel RU/EN processing.

This module provides language detection and routing capabilities to direct
audio segments to appropriate language-specific processing pipelines.
"""

import asyncio
from dataclasses import dataclass
import queue
import threading
import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from ..audio_io.language_id import SileroLanguageID
from .audio_data import RecognitionResult


@dataclass
class LanguageSegment:
    """Audio segment with detected language information.

    Attributes:
        audio: Audio samples (16kHz mono float32)
        language: Detected language code ('ru' or 'en')
        confidence: Detection confidence (0.0-1.0)
        timestamp: Segment timestamp for ordering
        speaker_id: Optional speaker ID from diarization
    """
    audio: NDArray[np.float32]
    language: Literal["ru", "en"] | None
    confidence: float
    timestamp: float
    speaker_id: str | None = None


class LanguageRouter:
    """Routes audio segments to language-specific processing queues.

    Uses Silero Language ID to detect speech language and route segments
    to appropriate RU or EN processing pipelines for parallel execution.
    """

    def __init__(
        self,
        lid_model: SileroLanguageID,
        ru_queue: queue.Queue[LanguageSegment],
        en_queue: queue.Queue[LanguageSegment],
        confidence_threshold: float = 0.7,
        default_language: Literal["ru", "en"] = "ru",
        shutdown_event: threading.Event | None = None,
    ):
        """Initialize language router.

        Args:
            lid_model: Silero Language ID model instance
            ru_queue: Queue for Russian segments
            en_queue: Queue for English segments
            confidence_threshold: Minimum confidence for language detection
            default_language: Default language when confidence is low
            shutdown_event: Event to signal shutdown
        """
        self.lid_model = lid_model
        self.ru_queue = ru_queue
        self.en_queue = en_queue
        self.confidence_threshold = confidence_threshold
        self.default_language = default_language
        self.shutdown_event = shutdown_event or threading.Event()

        # Statistics
        self.total_segments = 0
        self.ru_segments = 0
        self.en_segments = 0
        self.uncertain_segments = 0

    def route_segment(
        self,
        audio: NDArray[np.float32],
        timestamp: float,
        speaker_id: str | None = None,
    ) -> LanguageSegment:
        """Route a single audio segment to appropriate queue.

        Args:
            audio: Audio samples (16kHz mono float32)
            timestamp: Segment timestamp
            speaker_id: Optional speaker ID from diarization

        Returns:
            LanguageSegment with routing information
        """
        # Detect language
        lang_code, confidence = self.lid_model.detect(audio)

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            logger.warning(
                f"Low language detection confidence ({confidence:.3f}), "
                f"using default language: {self.default_language}"
            )
            lang_code = self.default_language
            self.uncertain_segments += 1

        # Map to supported languages
        if lang_code not in ("ru", "en"):
            logger.warning(
                f"Unsupported language '{lang_code}', "
                f"routing to default: {self.default_language}"
            )
            lang_code = self.default_language

        # Create segment
        segment = LanguageSegment(
            audio=audio,
            language=lang_code,
            confidence=confidence,
            timestamp=timestamp,
            speaker_id=speaker_id,
        )

        # Route to appropriate queue
        if lang_code == "ru":
            self.ru_queue.put(segment)
            self.ru_segments += 1
            logger.info(f"Routed segment to RU branch (confidence: {confidence:.3f})")
        else:
            self.en_queue.put(segment)
            self.en_segments += 1
            logger.info(f"Routed segment to EN branch (confidence: {confidence:.3f})")

        self.total_segments += 1

        return segment

    def get_statistics(self) -> dict[str, int]:
        """Get routing statistics.

        Returns:
            Dictionary with routing stats
        """
        return {
            "total": self.total_segments,
            "ru": self.ru_segments,
            "en": self.en_segments,
            "uncertain": self.uncertain_segments,
        }

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self.total_segments = 0
        self.ru_segments = 0
        self.en_segments = 0
        self.uncertain_segments = 0


class LanguageRouterThread:
    """Thread-based language router for continuous audio processing.

    Runs in a separate thread to continuously route audio segments
    from input queue to language-specific queues.
    """

    def __init__(
        self,
        input_queue: queue.Queue[RecognitionResult],
        lid_model: SileroLanguageID,
        ru_queue: queue.Queue[LanguageSegment],
        en_queue: queue.Queue[LanguageSegment],
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
        confidence_threshold: float = 0.7,
        default_language: Literal["ru", "en"] = "ru",
    ):
        """Initialize router thread.

        Args:
            input_queue: Input queue with audio segments
            lid_model: Language ID model
            ru_queue: Output queue for Russian
            en_queue: Output queue for English
            shutdown_event: Shutdown signal
            pause_time: Sleep time between queue checks
            confidence_threshold: Min confidence for detection
            default_language: Default when confidence is low
        """
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

        self.router = LanguageRouter(
            lid_model=lid_model,
            ru_queue=ru_queue,
            en_queue=en_queue,
            confidence_threshold=confidence_threshold,
            default_language=default_language,
            shutdown_event=shutdown_event,
        )

    def run(self) -> None:
        """Run the router thread loop."""
        logger.info("Language Router thread started")

        while not self.shutdown_event.is_set():
            try:
                # Get recognition result from input queue
                result = self.input_queue.get(timeout=self.pause_time)

                if not result.text:
                    logger.debug("Skipping empty recognition result")
                    continue

                # Note: In actual implementation, we'd need audio samples here
                # This is a simplified version - in real usage, RecognitionResult
                # would need to include audio samples or we'd need a separate
                # audio segment queue

                logger.debug(f"Processing segment: '{result.text[:50]}...'")

                # For now, just log - actual routing would happen with audio
                # This would be called from speech_listener with actual audio
                logger.warning(
                    "LanguageRouterThread needs audio samples - "
                    "should be integrated into speech_listener flow"
                )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in language router: {e}")
                logger.exception(e)
                time.sleep(self.pause_time)

        logger.info("Language Router thread finished")
