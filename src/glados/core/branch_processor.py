"""Language-specific branch processors for parallel ASR→TTS pipeline.

This module provides parallel processing pipelines for Russian and English
speech recognition and synthesis.
"""

import queue
import threading
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from ..ASR import TranscriberProtocol
from ..TTS import SpeechSynthesizerProtocol
from ..utils import spoken_text_converter as stc
from .audio_data import AudioMessage, RecognitionResult
from .language_router import LanguageSegment


class BranchProcessor:
    """Language-specific processing branch (ASR → Text Norm → TTS).

    Processes audio segments through language-specific ASR and TTS models,
    running in parallel with other language branches.
    """

    def __init__(
        self,
        language: str,
        input_queue: queue.Queue[LanguageSegment],
        output_queue: queue.Queue[AudioMessage],
        asr_model: TranscriberProtocol,
        tts_model: SpeechSynthesizerProtocol,
        stc_instance: stc.SpokenTextConverter,
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
        llm_queue: queue.Queue[RecognitionResult] | None = None,
        forward_to_llm: bool = False,
        generate_tts: bool = True,
    ):
        """Initialize branch processor.

        Args:
            language: Language code ('ru' or 'en')
            input_queue: Input queue with audio segments
            output_queue: Output queue for synthesized audio
            asr_model: ASR model for this language
            tts_model: TTS model for this language
            stc_instance: Text normalization converter
            shutdown_event: Shutdown signal
            pause_time: Sleep time between queue checks
            llm_queue: Optional queue to forward transcripts to the LLM pipeline
            forward_to_llm: Forward recognized text to LLM queue when True
            generate_tts: Whether to synthesize TTS for recognized text
        """
        self.language = language.upper()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.stc = stc_instance
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time
        self.llm_queue = llm_queue
        self.forward_to_llm = forward_to_llm
        self.generate_tts = generate_tts

        # Performance metrics
        self.segments_processed = 0
        self.total_asr_time = 0.0
        self.total_tts_time = 0.0
        self.errors = 0

    def run(self) -> None:
        """Run the processor thread loop."""
        logger.info(f"{self.language}-Branch Processor started")

        while not self.shutdown_event.is_set():
            try:
                # Get segment from input queue
                segment = self.input_queue.get(timeout=self.pause_time)

                if segment.audio is None or len(segment.audio) == 0:
                    logger.warning(f"{self.language}: Received empty audio segment")
                    continue

                # Process segment
                self._process_segment(segment)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{self.language}-Branch: Error processing segment: {e}")
                logger.exception(e)
                self.errors += 1
                time.sleep(self.pause_time)

        logger.info(f"{self.language}-Branch Processor finished")

    def _process_segment(self, segment: LanguageSegment) -> None:
        """Process a single audio segment through ASR→TTS pipeline.

        Args:
            segment: Audio segment with language info
        """
        try:
            # Step 1: ASR (Speech Recognition)
            asr_start = time.time()

            audio = segment.audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            logger.debug(
                f"{self.language}: Running ASR on {len(audio)/16000:.2f}s audio"
            )

            emotions = segment.emotions
            text = ""
            try:
                if hasattr(self.asr_model, "transcribe_with_emotions"):
                    text, emotions = self.asr_model.transcribe_with_emotions(audio)
                else:
                    text = self.asr_model.transcribe(audio)
            finally:
                asr_time = time.time() - asr_start

            text = (text or "").strip()

            if text:
                self.total_asr_time += asr_time
                logger.success(
                    f"{self.language}-ASR ({asr_time:.3f}s): '{text}'"
                )
            else:
                fallback = (segment.transcript or "").strip()
                if fallback:
                    logger.warning(
                        f"{self.language}: Branch ASR empty, falling back to routed transcript '{fallback}'"
                    )
                    text = fallback
                else:
                    logger.warning(f"{self.language}: No transcript available after ASR")
                    return

            # Forward transcript to LLM pipeline if configured
            if self.forward_to_llm and self.llm_queue is not None:
                recognition = RecognitionResult(
                    text=text,
                    emotions=emotions,
                    audio=segment.audio,
                    timestamp=segment.timestamp,
                    duration=segment.duration,
                    sample_rate=segment.sample_rate,
                    language=self.language.lower(),
                )
                try:
                    self.llm_queue.put_nowait(recognition)
                except queue.Full:
                    self.llm_queue.put(recognition)
                logger.info(
                    f"{self.language}-Branch: Forwarded transcript to LLM queue: '{text}'"
                )

            if not self.generate_tts:
                self.segments_processed += 1
                logger.debug(
                    f"{self.language}-Branch: generate_tts disabled, skipping TTS synthesis"
                )
                return

            # Step 2: Text Normalization
            normalized_text = self.stc.text_to_spoken(text)
            logger.debug(
                f"{self.language}: Normalized text: '{normalized_text}'"
            )

            # Step 3: TTS (Speech Synthesis)
            tts_start = time.time()

            audio_data = self.tts_model.generate_speech_audio(normalized_text)

            tts_time = time.time() - tts_start
            self.total_tts_time += tts_time

            if audio_data is None or len(audio_data) == 0:
                logger.warning(f"{self.language}: TTS returned empty audio")
                return

            logger.success(
                f"{self.language}-TTS ({tts_time:.3f}s): "
                f"Generated {len(audio_data)/self.tts_model.sample_rate:.2f}s audio"
            )

            # Step 4: Send to output mixer
            output_msg = AudioMessage(
                audio=audio_data,
                text=normalized_text,
                is_eos=False,
                sequence_num=self.segments_processed,
                language=self.language.lower(),
                timestamp=segment.timestamp,
            )

            self.output_queue.put(output_msg)

            self.segments_processed += 1

            total_time = asr_time + tts_time
            logger.info(
                f"{self.language}-Branch: Segment processed in {total_time:.3f}s "
                f"(ASR: {asr_time:.3f}s, TTS: {tts_time:.3f}s)"
            )

        except Exception as e:
            logger.error(
                f"{self.language}-Branch: Failed to process segment: {e}"
            )
            logger.exception(e)
            self.errors += 1

    def get_statistics(self) -> dict[str, float]:
        """Get processor performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_asr = (
            self.total_asr_time / self.segments_processed
            if self.segments_processed > 0
            else 0.0
        )
        avg_tts = (
            self.total_tts_time / self.segments_processed
            if self.segments_processed > 0
            else 0.0
        )

        return {
            "segments_processed": self.segments_processed,
            "total_asr_time": self.total_asr_time,
            "total_tts_time": self.total_tts_time,
            "avg_asr_time": avg_asr,
            "avg_tts_time": avg_tts,
            "errors": self.errors,
        }


# Convenience function to create and start branch processor threads
def create_branch_processors(
    ru_input_queue: queue.Queue[LanguageSegment],
    en_input_queue: queue.Queue[LanguageSegment],
    output_queue: queue.Queue[AudioMessage],
    ru_asr_model: TranscriberProtocol,
    en_asr_model: TranscriberProtocol,
    ru_tts_model: SpeechSynthesizerProtocol,
    en_tts_model: SpeechSynthesizerProtocol,
    stc_instance: stc.SpokenTextConverter,
    shutdown_event: threading.Event,
    pause_time: float = 0.05,
    llm_queue: queue.Queue[RecognitionResult] | None = None,
    forward_to_llm_languages: set[str] | None = None,
    tts_languages: set[str] | None = None,
) -> tuple[BranchProcessor, BranchProcessor]:
    """Create RU and EN branch processors.

    Args:
        ru_input_queue: Russian segment queue
        en_input_queue: English segment queue
        output_queue: Shared output queue
        ru_asr_model: Russian ASR model
        en_asr_model: English ASR model
        ru_tts_model: Russian TTS model
        en_tts_model: English TTS model
        stc_instance: Text converter
        shutdown_event: Shutdown signal
        pause_time: Loop pause time
        llm_queue: Shared queue for forwarding transcripts to LLM
        forward_to_llm_languages: Set of languages that should forward transcripts
        tts_languages: Languages for which TTS synthesis should run

    Returns:
        Tuple of (RU processor, EN processor)
    """
    if forward_to_llm_languages is None:
        forward_set: set[str] = set()
    else:
        forward_set = {lang.lower() for lang in forward_to_llm_languages}

    if tts_languages is None:
        tts_source = {"ru", "en"}
    else:
        tts_source = tts_languages

    tts_set = {lang.lower() for lang in tts_source}

    ru_processor = BranchProcessor(
        language="ru",
        input_queue=ru_input_queue,
        output_queue=output_queue,
        asr_model=ru_asr_model,
        tts_model=ru_tts_model,
        stc_instance=stc_instance,
        shutdown_event=shutdown_event,
        pause_time=pause_time,
        llm_queue=llm_queue,
        forward_to_llm="ru" in forward_set,
        generate_tts="ru" in tts_set,
    )

    en_processor = BranchProcessor(
        language="en",
        input_queue=en_input_queue,
        output_queue=output_queue,
        asr_model=en_asr_model,
        tts_model=en_tts_model,
        stc_instance=stc_instance,
        shutdown_event=shutdown_event,
        pause_time=pause_time,
        llm_queue=llm_queue,
        forward_to_llm="en" in forward_set,
        generate_tts="en" in tts_set,
    )

    return ru_processor, en_processor
