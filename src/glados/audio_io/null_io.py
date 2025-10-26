"""Headless audio backend for environments without sound hardware."""

from __future__ import annotations

import queue
from loguru import logger
import numpy as np
from numpy.typing import NDArray


class NullAudioIO:
    """Audio backend that disables real I/O while keeping the pipeline alive."""

    SAMPLE_RATE: int = 16000

    def __init__(self, vad_threshold: float | None = None) -> None:
        self.vad_threshold = 0.0 if vad_threshold is None else vad_threshold
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        logger.warning("NullAudioIO active: audio input/output is disabled.")

    def start_listening(self) -> None:
        logger.info("NullAudioIO: start_listening() called - no audio will be captured.")

    def stop_listening(self) -> None:
        logger.info("NullAudioIO: stop_listening() called.")

    def start_speaking(
        self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = ""
    ) -> None:
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE
        logger.info(
            "NullAudioIO: start_speaking() called - suppressing playback (len=%s, sr=%s).",
            len(audio_data),
            sample_rate,
        )

    def measure_percentage_spoken(
        self, total_samples: int, sample_rate: int | None = None
    ) -> tuple[bool, int]:
        return False, 100

    def check_if_speaking(self) -> bool:
        return False

    def stop_speaking(self) -> None:
        logger.info("NullAudioIO: stop_speaking() called.")

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        return self._sample_queue

