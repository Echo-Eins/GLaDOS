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
    MIN_VOICE_ACTIVITY_RATIO: float = 0.20  # Minimum fraction of VAD-positive frames required to treat audio as speech

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
        1. Checks if accumulated samples contain actual speech (not just noise/silence)
        2. Transcribes the collected audio samples using the ASR model.
        3. If transcription is successful:
            a. Checks for the `wake_word` (if configured).
            b. If the wake word is detected (or not required), the transcribed text is
               placed into the `llm_queue`, and `processing_active_event` is set.
        4. Resets the listener's internal state using `self.reset()`, preparing for the next input.
        """
        logger.debug("Detected pause after speech. Processing...")

        # Check if samples contain actual speech energy (not just silence/noise)
        if not self._samples:
            logger.warning("No samples collected, skipping ASR")
            self.reset()
            return

        audio = np.concatenate(self._samples)
        rms_energy = np.sqrt(np.mean(audio**2))
        duration_s = len(audio) / 16000

        # Always log RMS for debugging
        logger.info(f"RMS energy: {rms_energy:.6f} | Samples: {len(self._samples)} | Total duration: {duration_s:.2f}s")

        # Threshold: if RMS is too low, this is likely a false VAD trigger
        if rms_energy < 0.01:  # Adjust threshold based on your microphone
            logger.warning(f"🚫 Ignoring detected audio: RMS energy too low ({rms_energy:.6f} < 0.01), likely false VAD trigger")
            self.reset()
            return

        # Check minimum duration to avoid processing very short audio segments
        # that are likely false triggers, echoes, or incomplete speech
        MIN_DURATION_S = 0.3
        if duration_s < MIN_DURATION_S:
            logger.warning(
                f"🚫 Ignoring detected audio: Duration too short ({duration_s:.2f}s < {MIN_DURATION_S}s). "
                f"This may be echo, noise, or incomplete speech. Speak longer phrases."
            )
            self.reset()
            return

        # Require a sufficient proportion of frames with positive VAD confidence.
        # This helps suppress persistent false activations triggered by echo or noise.
        #
        # CRITICAL: We must ONLY count frames in the active speech region, excluding:
        # 1. Pre-roll buffer (first ~25 frames from _buffer, mostly False)
        # 2. Trailing silence (last ~20 frames waiting for PAUSE_LIMIT, all False)
        #
        # Calculate the active speech region: from first True VAD to last True VAD
        total_frames = len(self._vad_flags)

        # Find the range of actual speech activity
        first_voiced_idx = next((i for i, flag in enumerate(self._vad_flags) if flag), None)
        last_voiced_idx = next((i for i, flag in enumerate(reversed(self._vad_flags)) if flag), None)

        if first_voiced_idx is None or last_voiced_idx is None:
            # No voice activity detected at all
            logger.warning(
                "🚫 Ignoring detected audio: no voice activity detected in {} frames",
                total_frames
            )
            self.reset()
            return

        # Convert last_voiced_idx from reversed index to forward index
        last_voiced_idx = total_frames - 1 - last_voiced_idx

        # Extract active region (from first voiced to last voiced, inclusive)
        active_vad_flags = self._vad_flags[first_voiced_idx:last_voiced_idx + 1]
        active_speaking_flags = self._speaking_flags[first_voiced_idx:last_voiced_idx + 1]

        # Filter out frames that happened while the assistant was actively speaking.
        # Those frames are usually echo from our own TTS output and should not be
        # counted towards the user's speech statistics.
        user_frame_mask = [not speaking for speaking in active_speaking_flags]
        user_active_frames = sum(user_frame_mask)
        voiced_frames = sum(
            1 for flag, keep in zip(active_vad_flags, user_frame_mask) if keep and flag
        )

        # Fall back to the raw active-frame count when all frames happened while the
        # assistant was speaking (e.g. immediate wake word after TTS interruption).
        effective_denominator = user_active_frames or len(active_vad_flags)
        voice_ratio = (voiced_frames / effective_denominator) if effective_denominator else 0.0

        logger.debug(
            f"Voice activity stats | total_frames={total_frames} | "
            f"active_region=[{first_voiced_idx}:{last_voiced_idx+1}] ({len(active_vad_flags)} frames) | "
            f"voiced_frames={voiced_frames} | ratio={voice_ratio:.2%}"
        )

        if voice_ratio < self.MIN_VOICE_ACTIVITY_RATIO:
            logger.warning(
                "🚫 Ignoring detected audio: voice activity ratio too low ({:.0f}% < {:.0f}%). "
                "Likely echo or background noise.",
                voice_ratio * 100,
                self.MIN_VOICE_ACTIVITY_RATIO * 100,
            )
            self.reset()
            return

        # Discard segments where all voiced frames were captured while the assistant itself was speaking.
        # These are typically acoustic echoes that slipped through the interrupt logic.
        # Also limit this check to the active speech region only.
        active_vad_flags_in_region = active_vad_flags
        active_speaking_flags_in_region = active_speaking_flags

        voiced_frames_during_speech = sum(
            1 for vad_flag, speaking_flag in zip(active_vad_flags_in_region, active_speaking_flags_in_region)
            if vad_flag and speaking_flag
        )
        user_voiced_frames = voiced_frames - voiced_frames_during_speech

        if voiced_frames and user_voiced_frames <= 0:
            logger.warning(
                "🚫 Ignoring detected audio: all voice activity occurred while assistant speech was active. "
                "Most likely acoustic echo."
            )
            self.reset()
            return

        # Only feed the active speech portion to the ASR to avoid long silences
        # or assistant echo contaminating the recognition request.
        active_samples = self._samples[first_voiced_idx:last_voiced_idx + 1]
        recognition = self.asr(active_samples)

        if recognition.text:
            logger.success(f"ASR text: '{recognition.text}'")
            if recognition.emotions:
                logger.success(f"ASR emotions: {dict(recognition.emotions)}")

            if self.wake_word and not self._wakeword_detected(recognition.text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                self.llm_queue.put(recognition)
                self.processing_active_event.set()
        else:
            # ANOMALY: High RMS but empty ASR result - real speech may have been lost!
            if rms_energy >= 0.01:
                logger.error(
                    f"⚠️ ANOMALY DETECTED: RMS energy was high ({rms_energy:.6f} >= 0.01) "
                    f"but ASR returned empty text! Duration: {duration_s:.2f}s. "
                    f"This may indicate: 1) Echo/noise mistaken for speech, 2) Audio quality issues, "
                    f"or 3) Speech too unclear to recognize. Try speaking louder and more clearly."
                )

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

        # Check audio amplitude and signal strength
        max_abs_val = np.max(np.abs(audio))
        rms_val = np.sqrt(np.mean(audio**2))
        logger.debug(f"ASR input | max_amplitude={max_abs_val:.6f}, RMS={rms_val:.6f}, samples={len(audio)}")

        if max_abs_val < 1e-10:  # Threshold for effectively silent audio
            logger.warning("ASR received effectively silent audio")
            return RecognitionResult(text="", emotions=None)

        # CRITICAL: Check signal amplitude BEFORE normalization
        # Even if RMS is adequate, if max amplitude is very low, this is likely
        # background noise or echo that will produce garbage after normalization
        MIN_SIGNAL_AMPLITUDE = 0.01  # Minimum peak amplitude for valid speech
        if max_abs_val < MIN_SIGNAL_AMPLITUDE:
            logger.warning(
                f"🚫 ASR: Signal amplitude too low (max={max_abs_val:.6f} < {MIN_SIGNAL_AMPLITUDE}). "
                f"This is likely background noise or echo, not real speech. Skipping ASR to avoid NaN."
            )
            return RecognitionResult(text="", emotions=None)

        # Additional check: peak-to-RMS ratio (crest factor)
        # Real speech typically has crest factor of 3-6
        # Pure noise/echo often has abnormal ratios (very high or very low)
        crest_factor = max_abs_val / rms_val if rms_val > 1e-10 else 0
        logger.debug(f"ASR crest factor (peak/RMS): {crest_factor:.2f}")

        # If crest factor is abnormally high (>20), it's likely a spike/click, not speech
        MAX_CREST_FACTOR = 20.0
        if crest_factor > MAX_CREST_FACTOR:
            logger.warning(
                f"🚫 ASR: Abnormal crest factor ({crest_factor:.2f} > {MAX_CREST_FACTOR}). "
                f"Signal contains spikes/clicks, not speech. Skipping ASR to avoid NaN."
            )
            return RecognitionResult(text="", emotions=None)

        # Normalize to full range [-1.0, 1.0]
        audio = audio / max_abs_val

        emotions = None

        if hasattr(self.asr_model, "transcribe_with_emotions"):
            detected_text, emotions = getattr(self.asr_model, "transcribe_with_emotions")(audio)
            # Sanitize emotions to remove NaN/inf values that would break JSON serialization
            emotions = sanitize_emotions(emotions)
        else:
            detected_text = self.asr_model.transcribe(audio)

        # Strip whitespace and check if result is empty
        detected_text = (detected_text or "").strip()

        if not detected_text:
            logger.warning("ASR produced empty or whitespace-only transcription")
            return RecognitionResult(text="", emotions=emotions)

        return RecognitionResult(text=detected_text, emotions=emotions)
