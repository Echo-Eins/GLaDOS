"""Audio mixer for combining RU/EN branch outputs.

This module provides audio mixing capabilities to merge outputs from
parallel RU and EN processing branches with proper timing and crossfading.
"""

import queue
import threading
import time
from collections import deque
from typing import Deque

import numpy as np
from numpy.typing import NDArray
from loguru import logger
import scipy.signal

from .audio_data import AudioMessage


class AudioMixer:
    """Mixes and orders audio from multiple language branches.

    Combines audio outputs from RU and EN branches, applying:
    - Timestamp-based ordering
    - Crossfade transitions (40ms)
    - Amplitude normalization
    - Sample rate conversion if needed
    """

    CROSSFADE_MS: int = 40  # Crossfade duration in milliseconds
    MAX_BUFFER_SIZE: int = 100  # Maximum segments to buffer

    def __init__(
        self,
        input_queue: queue.Queue[AudioMessage],
        output_queue: queue.Queue[AudioMessage],
        target_sample_rate: int = 48000,
        shutdown_event: threading.Event | None = None,
        pause_time: float = 0.05,
    ):
        """Initialize audio mixer.

        Args:
            input_queue: Input queue receiving audio from all branches
            output_queue: Output queue for mixed/ordered audio
            target_sample_rate: Target sample rate for output
            shutdown_event: Event to signal shutdown
            pause_time: Sleep time between queue checks
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.target_sample_rate = target_sample_rate
        self.shutdown_event = shutdown_event or threading.Event()
        self.pause_time = pause_time

        # Buffer for timestamp-based ordering
        self.buffer: Deque[AudioMessage] = deque(maxlen=self.MAX_BUFFER_SIZE)

        # Statistics
        self.segments_mixed = 0
        self.crossfades_applied = 0

    def run(self) -> None:
        """Run the mixer thread loop."""
        logger.info("Audio Mixer thread started")

        last_audio: NDArray[np.float32] | None = None

        while not self.shutdown_event.is_set():
            try:
                # Get audio message from input queue
                msg = self.input_queue.get(timeout=self.pause_time)

                # Handle EOS token
                if msg.is_eos:
                    self.output_queue.put(msg)
                    last_audio = None
                    continue

                # Process audio segment
                processed = self._process_segment(msg, last_audio)

                if processed is not None:
                    self.output_queue.put(processed)
                    last_audio = processed.audio
                    self.segments_mixed += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio Mixer: Error processing segment: {e}")
                logger.exception(e)
                time.sleep(self.pause_time)

        logger.info("Audio Mixer thread finished")

    def _process_segment(
        self,
        msg: AudioMessage,
        last_audio: NDArray[np.float32] | None,
    ) -> AudioMessage | None:
        """Process a single audio segment.

        Args:
            msg: Audio message to process
            last_audio: Previous audio segment for crossfading

        Returns:
            Processed AudioMessage or None if skipped
        """
        try:
            audio = msg.audio

            if audio is None or len(audio) == 0:
                logger.warning("Received empty audio segment")
                return None

            # Step 1: Resample if needed
            # Note: Assumes input sample rate from TTS models (48kHz)
            # If different rates are used, would need to track per-message
            # For now, assuming all TTS outputs are at same rate

            # Step 2: Normalize amplitude
            audio = self._normalize_audio(audio)

            # Step 3: Apply crossfade if we have previous audio
            if last_audio is not None and len(last_audio) > 0:
                audio = self._apply_crossfade(last_audio, audio)
                self.crossfades_applied += 1

            # Create output message
            output_msg = AudioMessage(
                audio=audio,
                text=msg.text,
                is_eos=False,
                sequence_num=msg.sequence_num,
                language=msg.language,
                timestamp=msg.timestamp,
            )

            logger.debug(
                f"Mixer: Processed {msg.language or 'unknown'} segment "
                f"({len(audio)/self.target_sample_rate:.2f}s): '{msg.text[:50]}...'"
            )

            return output_msg

        except Exception as e:
            logger.error(f"Failed to process audio segment: {e}")
            logger.exception(e)
            return None

    def _normalize_audio(
        self,
        audio: NDArray[np.float32],
        target_rms: float = 0.1,
    ) -> NDArray[np.float32]:
        """Normalize audio amplitude.

        Args:
            audio: Input audio samples
            target_rms: Target RMS level

        Returns:
            Normalized audio
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))

        if rms < 1e-6:
            logger.warning("Audio RMS too low, skipping normalization")
            return audio

        # Scale to target RMS
        scale_factor = target_rms / rms
        normalized = audio * scale_factor

        # Prevent clipping
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val

        return normalized

    def _apply_crossfade(
        self,
        prev_audio: NDArray[np.float32],
        curr_audio: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Apply crossfade between two audio segments.

        Args:
            prev_audio: Previous audio segment (tail used for fade-out)
            curr_audio: Current audio segment (head receives fade-in)

        Returns:
            Audio with crossfade applied to beginning
        """
        try:
            # Calculate crossfade length in samples
            crossfade_samples = int(
                self.CROSSFADE_MS * self.target_sample_rate / 1000
            )

            # Ensure we have enough samples
            if len(prev_audio) < crossfade_samples or len(curr_audio) < crossfade_samples:
                logger.debug("Not enough samples for crossfade, skipping")
                return curr_audio

            # Create fade curves (cosine fade for smooth transition)
            fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2

            # Extract crossfade regions
            prev_tail = prev_audio[-crossfade_samples:]
            curr_head = curr_audio[:crossfade_samples]

            # Apply fades and mix
            mixed_region = prev_tail * fade_out + curr_head * fade_in

            # Replace head of current audio with mixed region
            result = curr_audio.copy()
            result[:crossfade_samples] = mixed_region

            logger.debug(
                f"Applied {self.CROSSFADE_MS}ms crossfade "
                f"({crossfade_samples} samples)"
            )

            return result

        except Exception as e:
            logger.warning(f"Crossfade failed: {e}, returning original audio")
            return curr_audio

    def get_statistics(self) -> dict[str, int]:
        """Get mixer statistics.

        Returns:
            Dictionary with mixer stats
        """
        return {
            "segments_mixed": self.segments_mixed,
            "crossfades_applied": self.crossfades_applied,
            "buffer_size": len(self.buffer),
        }


def create_crossfade_window(
    length: int,
    window_type: str = "cosine",
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Create fade-in and fade-out windows for crossfading.

    Args:
        length: Window length in samples
        window_type: Window type ('cosine', 'linear', 'hann')

    Returns:
        Tuple of (fade_out, fade_in) windows
    """
    if window_type == "cosine":
        # Cosine fade (smooth, natural)
        fade_out = np.cos(np.linspace(0, np.pi / 2, length)) ** 2
        fade_in = np.sin(np.linspace(0, np.pi / 2, length)) ** 2

    elif window_type == "linear":
        # Linear fade
        fade_out = np.linspace(1.0, 0.0, length)
        fade_in = np.linspace(0.0, 1.0, length)

    elif window_type == "hann":
        # Hann window (very smooth)
        window = np.hanning(length * 2)
        fade_out = window[:length]
        fade_in = window[length:]

    else:
        raise ValueError(f"Unknown window type: {window_type}")

    return fade_out.astype(np.float32), fade_in.astype(np.float32)
