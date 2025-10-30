"""
Speech listener module for the Glados voice assistant.

This module provides the SpeechListener class that handles audio input streaming,
voice activity detection, speech recognition, and wake word detection.
"""

from collections import deque
import math
import queue
import threading
import time

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from ..ASR import TranscriberProtocol
from ..audio_io import AudioProtocol
from .audio_data import RecognitionResult


def sanitize_emotions(emotions: dict[str, float] | None) -> dict[str, float] | None:
    """Sanitize emotion dictionary by replacing NaN/inf values with 0.0.

    Args:
        emotions: Dictionary mapping emotion names to probability values

    Returns:
        Sanitized emotions dictionary or None if input is None
    """
    if emotions is None:
        return None

    sanitized = {}
    has_invalid = False
    for key, value in emotions.items():
        if not isinstance(value, (int, float)):
            sanitized[key] = 0.0
            has_invalid = True
        elif math.isnan(value) or math.isinf(value):
            sanitized[key] = 0.0
            has_invalid = True
        else:
            sanitized[key] = float(value)

    if has_invalid:
        logger.warning(f"Sanitized invalid emotion values (NaN/inf): {emotions} -> {sanitized}")

    return sanitized


class SpeechListener:
    """
    Manages audio input and speech processing for a voice assistant.

    This class handles capturing audio, performing Voice Activity Detection (VAD),
    buffering pre-activation audio, triggering Automatic Speech Recognition (ASR),
    and coordinating with Language Model (LLM) and Text-to-Speech (TTS) components
    via shared events and queues. It supports optional wake word detection.
    """

    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity
    ECHO_GRACE_PERIOD: float = 0.8  # Seconds to ignore VAD after speaking ends (acoustic echo suppression)

    def __init__(
        self,
        audio_io: AudioProtocol,  # Replace with actual type if known
        llm_queue: queue.Queue[RecognitionResult],
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,
        asr_model: TranscriberProtocol,
        wake_word: str | None,
        pause_time: float,
        interruptible: bool = True,
    ) -> None:
        """
        Initializes the SpeechListener with audio I/O, inter-thread communication, and ASR model.

        Args:
            audio_io: An instance conforming to `AudioProtocol` for audio input/output.
            llm_queue: A queue for sending transcribed text to the language model.
            shutdown_event: A threading.Event to signal the application to shut down.
            currently_speaking_event: A threading.Event indicating if the assistant is currently speaking.
            processing_active_event: A threading.Event indicating if processing is active (e.g., for LLM/TTS).
            asr_model: An instance conforming to `TranscriberProtocol` for speech recognition.
            wake_word: Optional wake word string to activate the assistant. Defaults to None.
            interruptible: If True, allows new speech input to interrupt ongoing assistant speech.
        """
        self.audio_io = audio_io
        self.llm_queue = llm_queue
        self.asr_model = asr_model
        self.wake_word = wake_word.lower() if wake_word else None
        self.pause_time = pause_time
        self.interruptible = interruptible

        ## Circular buffer to hold pre-activation samples
        #self._buffer: deque[NDArray[np.float32]] = deque(maxlen=self.BUFFER_SIZE // self.VAD_SIZE)

        # Circular buffer to hold pre-activation samples alongside their VAD decisions
        self._buffer: deque[tuple[NDArray[np.float32], bool, bool]] = deque(
            maxlen=self.BUFFER_SIZE // self.VAD_SIZE
        )

        self._sample_queue = self.audio_io.get_sample_queue()

        # Internal state variables
        self._recording_started = False
        self._samples: list[NDArray[np.float32]] = []
        self._vad_flags: list[bool] = []
        self._speaking_flags: list[bool] = []
        self._gap_counter = 0

        # Echo suppression: track when assistant stopped speaking
        self._last_speaking_end_time: float = 0.0
        self._was_speaking: bool = False

        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event

    def run(self) -> None:
        """
        Starts the main listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters a loop that
        listens for incoming audio samples and their Voice Activity Detection (VAD) confidence.
        It retrieves samples from an internal queue and processes them via `_handle_audio_sample`.
        The loop runs until the `shutdown_event` is set. It also handles brief pauses
        in audio input using a timeout.

        Raises:
            Exception: Catches and logs general exceptions encountered during the listening loop,
                       without stopping the loop unless `shutdown_event` is set.
        """
        logger.success("Audio Modules Operational")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                try:
                    # Use a timeout for the queue get
                    sample, vad_confidence = self._sample_queue.get(timeout=self.pause_time)
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    # Timeout occurred, loop again to check shutdown_event
                    continue
                except (OSError, RuntimeError) as e:  # More specific exceptions
                    if not self.shutdown_event.is_set():  # Only log if not shutting down
                        logger.error(f"Error in listen loop ({type(e).__name__}): {e}")
                    continue

            logger.info("Shutdown event detected in listen loop, exiting loop.")

        finally:
            self.audio_io.stop_listening()
            logger.info("Listen event loop is stopping/exiting.")

        logger.info("Speech Listener thread finished.")

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Routes the processing of an individual audio sample based on the current recording state.

        If recording has not started, the sample contributes to the pre-activation buffer.
        Once recording is active, the sample is added to the main speech segment
        and contributes to the voice activity gap detection.

        Args:
            sample: The audio sample (numpy array) to process.
            vad_confidence: True if voice activity is detected in the sample, False otherwise.
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Manages the pre-activation circular buffer and handles voice activity detection.

        Samples are continuously added to a circular buffer until voice activity is detected.
        Upon VAD detection:
        - It checks for interruptibility if the assistant is currently speaking.
        - The assistant's speaking is stopped (`audio_io.stop_speaking()`).
        - The `processing_active_event` is cleared, pausing LLM/TTS activity.
        - The buffered samples are moved to `_samples`, and `_recording_started` is set to True.

        Args:
            sample: The current audio sample (numpy array) to be added to the buffer.
            vad_confidence: True if voice activity is detected in the sample, False otherwise.
        """
        # Track speaking state transitions to detect when echo grace period should start
        is_currently_speaking = self.currently_speaking_event.is_set()
        if self._was_speaking and not is_currently_speaking:
            # Transition from speaking to not speaking - start grace period
            self._last_speaking_end_time = time.time()
            logger.debug(f"Assistant stopped speaking, starting {self.ECHO_GRACE_PERIOD}s echo grace period")
        self._was_speaking = is_currently_speaking

        if vad_confidence:
            # Check if we're within grace period after assistant stopped speaking
            time_since_speaking_ended = time.time() - self._last_speaking_end_time
            if time_since_speaking_ended < self.ECHO_GRACE_PERIOD:
                logger.debug(f"VAD detected but within echo grace period ({time_since_speaking_ended:.2f}s < {self.ECHO_GRACE_PERIOD}s), ignoring")
                # CRITICAL: Still add to buffer to maintain continuity, but don't trigger recording
                self._buffer.append((sample, False, is_currently_speaking))  # Mark as non-voice to prevent false activation
                return

            if not self.interruptible and self.currently_speaking_event.is_set():
                logger.debug(f"Detected voice activity but interruptibility is disabled: {self.interruptible=}, {self.currently_speaking_event.is_set()=}")
                # CRITICAL: Still add to buffer to maintain continuity
                self._buffer.append((sample, False, is_currently_speaking))  # Mark as non-voice to prevent false activation
                return

            # VAD confidence is True and we're past grace period → activate recording
            # CRITICAL: Add current sample to buffer BEFORE copying, so it's included in recording
            self._buffer.append((sample, vad_confidence, is_currently_speaking))

            self.audio_io.stop_speaking()
            self.processing_active_event.clear()

            # Copy buffer contents to recording lists
            self._samples = [chunk for chunk, _, _ in self._buffer]
            self._vad_flags = [flag for _, flag, _ in self._buffer]
            self._speaking_flags = [speaking for _, _, speaking in self._buffer]

            self._recording_started = True
        else:
            # No voice activity detected → just add to buffer and continue
            self._buffer.append((sample, vad_confidence, is_currently_speaking))

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Accumulates audio samples and tracks pauses after voice activation.

        This method appends incoming audio samples to `self._samples`. It increments
        `_gap_counter` when no voice activity is detected. If the `_gap_counter`
        exceeds `PAUSE_LIMIT`, it signifies the end of a speech segment, triggering
        `_process_detected_audio`. Otherwise, if voice is detected, the gap counter is reset.

        Args:
            sample: A single audio sample (numpy array) from the input stream.
            vad_confidence: True if voice activity is currently detected, False otherwise.
        """
        self._samples.append(sample)
        self._vad_flags.append(vad_confidence)
        self._speaking_flags.append(self.currently_speaking_event.is_set())

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Checks if the transcribed text contains a sufficiently similar match to the configured wake word.

        This method iterates through words in the `text` and calculates the Levenshtein distance
        (edit distance) between each word (converted to lowercase) and the `wake_word`.
        A match is considered found if the `closest_distance` is less than `SIMILARITY_THRESHOLD`.
        This helps account for minor misrecognitions of the wake word.

        Args:
            text: The transcribed text string to check for wake word similarity.

        Returns:
            True if a word in the text matches the wake word within the similarity threshold, False otherwise.

        Raises:
            AssertionError: If `self.wake_word` is None.
        """
        if self.wake_word is None:
            raise ValueError("Wake word should not be None")

        words = text.split()
        closest_distance = min(distance(word.lower(), self.wake_word) for word in words)
        return closest_distance < self.SIMILARITY_THRESHOLD

    def reset(self) -> None:
        """
        Resets the internal state of the speech listener, clearing all audio buffers and counters.

        This prepares the listener for a new speech segment by:
        - Setting `_recording_started` to False.
        - Clearing the accumulated `_samples`.
        - Resetting the `_gap_counter`.
        - Emptying the pre-activation circular buffer (`_buffer.queue`), safely using its internal mutex.
        - Resetting VAD model state to prevent accumulated noise from affecting future detections.
        - CRITICAL: Clearing the sample queue to remove accumulated echo/noise from playback period.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()

        self._vad_flags.clear()
        self._speaking_flags.clear()

        self._gap_counter = 0
        self._buffer.clear()

        # CRITICAL: Clear the sample queue to remove samples accumulated during playback
        # During long GLaDOS responses, microphone continues capturing and queue fills with echo
        queue_size = self._sample_queue.qsize()
        if queue_size > 0:
            logger.debug(f"Clearing {queue_size} accumulated samples from queue (likely echo/noise during playback)")
            while not self._sample_queue.empty():
                try:
                    self._sample_queue.get_nowait()
                except queue.Empty:
                    break

        # CRITICAL: Reset VAD model state to clear accumulated context and internal state
        # This prevents noise/echo from previous segments from affecting future VAD decisions
        self.audio_io.reset_vad_state()

    def _process_detected_audio(self) -> None:
        """
        Processes the accumulated audio samples once a speech pause is detected.

        This method performs the following steps:
        1. Transcribes the collected audio samples using the ASR model.
        2. If transcription is successful:
            a. Checks for the `wake_word` (if configured).
            b. If the wake word is detected (or not required), the transcribed text is
               placed into the `llm_queue`, and `processing_active_event` is set.
        3. Resets the listener's internal state using `self.reset()`, preparing for the next input.
        """
        logger.debug("Detected pause after speech. Processing...")

        if not self._samples:
            logger.warning("No samples collected, skipping ASR")
            self.reset()
            return

        audio = np.concatenate(self._samples)
        duration_s = len(audio) / 16000
        logger.info(f"Processing {duration_s:.2f}s audio with {len(self._samples)} samples")

        # Echo detection: Discard segments where all voiced frames were captured while assistant was speaking
        # Find the range of actual speech activity
        total_frames = len(self._vad_flags)
        first_voiced_idx = next((i for i, flag in enumerate(self._vad_flags) if flag), None)
        last_voiced_idx = next((i for i, flag in enumerate(reversed(self._vad_flags)) if flag), None)

        if first_voiced_idx is not None and last_voiced_idx is not None:
            # Convert last_voiced_idx from reversed index to forward index
            last_voiced_idx = total_frames - 1 - last_voiced_idx

            # Extract active region (from first voiced to last voiced, inclusive)
            active_vad_flags = self._vad_flags[first_voiced_idx:last_voiced_idx + 1]
            active_speaking_flags = self._speaking_flags[first_voiced_idx:last_voiced_idx + 1]

            # Count voiced frames that occurred while assistant was speaking
            voiced_frames_during_speech = sum(
                1 for vad_flag, speaking_flag in zip(active_vad_flags, active_speaking_flags)
                if vad_flag and speaking_flag
            )
            voiced_frames = sum(active_vad_flags)
            user_voiced_frames = voiced_frames - voiced_frames_during_speech

            # If all voice activity was during assistant speech, it's likely echo
            if voiced_frames and user_voiced_frames <= 0:
                logger.warning(
                    "Ignoring detected audio: all voice activity occurred while assistant was speaking. "
                    "Most likely acoustic echo."
                )
                self.reset()
                return

        recognition = self.asr(self._samples)

        if recognition.text:
            logger.success(f"ASR: '{recognition.text}'")
            if recognition.emotions:
                logger.info(f"Emotions: {dict(recognition.emotions)}")

            if self.wake_word and not self._wakeword_detected(recognition.text):
                logger.info(f"Wake word '{self.wake_word}' not detected")
            else:
                self.llm_queue.put(recognition)
                self.processing_active_event.set()
        else:
            logger.warning("ASR returned empty transcription")

        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> RecognitionResult:
        """
        Performs Automatic Speech Recognition (ASR) on a list of audio samples.

        The samples are first concatenated into a single audio array. This combined
        audio is then normalized to a range of [-1.0, 1.0] to ensure consistent
        volume levels before being passed to the ASR model for transcription.

        Args:
            samples: A list of numpy arrays (float32) containing audio sample chunks.

        Returns:
            The transcribed text as a string.
        """
        if not samples:
            logger.warning("ASR received empty sample list")
            return RecognitionResult(text="", emotions=None)

        audio = np.concatenate(samples)
        max_abs_val = np.max(np.abs(audio))

        if max_abs_val < 1e-10:
            logger.warning("ASR received silent audio")
            return RecognitionResult(text="", emotions=None)

        # Normalize to full range [-1.0, 1.0]
        audio = audio / max_abs_val

        # Call ASR
        if hasattr(self.asr_model, "transcribe_with_emotions"):
            detected_text, emotions = self.asr_model.transcribe_with_emotions(audio)
            emotions = sanitize_emotions(emotions)
        else:
            detected_text = self.asr_model.transcribe(audio)
            emotions = None

        detected_text = (detected_text or "").strip()
        return RecognitionResult(text=detected_text, emotions=emotions)
