import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict

from loguru import logger
import numpy as np

from ..TTS import SpeechSynthesizerProtocol
from ..utils import spoken_text_converter as stc
from .audio_data import AudioMessage, TTSTextMessage


class TextToSpeechSynthesizer:
    """
    A thread that synthesizes text to speech using a TTS model and a spoken text converter.
    It reads text from a queue, processes it, generates audio, and puts the audio messages into an output queue.
    This class is designed to run in a separate thread, continuously checking for new text to
    synthesize until a shutdown event is set.
    """

    MAX_PARALLEL_JOBS: int = 3  # Process up to 3 paragraphs in parallel

    def __init__(
        self,
        tts_input_queue: queue.Queue[TTSTextMessage],
        audio_output_queue: queue.Queue[AudioMessage],
        tts_model: SpeechSynthesizerProtocol,
        stc_instance: stc.SpokenTextConverter,
        shutdown_event: threading.Event,
        pause_time: float,
    ) -> None:
        self.tts_input_queue = tts_input_queue
        self.audio_output_queue = audio_output_queue
        self.tts_model = tts_model
        self.stc = stc_instance
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

        # Thread pool for parallel TTS processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.MAX_PARALLEL_JOBS,
            thread_name_prefix="TTS-Worker"
        )

        # Track pending futures by sequence number
        self.pending_futures: Dict[int, Future] = {}

    def _process_text(self, text_msg: TTSTextMessage) -> AudioMessage:
        """
        Process a single text message through TTS pipeline.
        This method is executed in parallel by ThreadPoolExecutor.

        Args:
            text_msg: Text message with sequence number

        Returns:
            AudioMessage with synthesized audio and sequence number
        """
        try:
            start_time = time.time()
            spoken_text_variant = self.stc.text_to_spoken(text_msg.text)
            audio_data = self.tts_model.generate_speech_audio(spoken_text_variant)
            processing_time = time.time() - start_time

            audio_duration = len(audio_data) / self.tts_model.sample_rate if len(audio_data) > 0 else 0
            logger.info(
                f"TTS Worker #{text_msg.sequence_num}: Complete in {processing_time:.2f}s, "
                f"audio length: {audio_duration:.2f}s for text: '{spoken_text_variant[:50]}...'"
            )

            return AudioMessage(
                audio=audio_data,
                text=spoken_text_variant,
                is_eos=False,
                sequence_num=text_msg.sequence_num
            )

        except Exception as e:
            logger.error(f"TTS Worker #{text_msg.sequence_num}: Failed to generate speech: {e}")
            logger.exception(e)
            # Return empty audio on error
            return AudioMessage(
                audio=np.array([], dtype=np.float32),
                text=text_msg.text,
                is_eos=False,
                sequence_num=text_msg.sequence_num
            )

    def run(self) -> None:
        """
        Starts the main loop for the TTS Synthesizer thread with parallel processing.

        This method continuously checks the TTS input queue for text to synthesize.
        It submits text messages to a ThreadPool for parallel TTS processing (up to 3 at once).
        Results are sent to the audio output queue preserving sequence order.

        The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("TextToSpeechSynthesizer thread started (parallel mode: up to 3 jobs)")

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get next text message from queue
                    text_msg = self.tts_input_queue.get(timeout=self.pause_time)

                    # Handle EOS token
                    if text_msg.is_eos:
                        logger.debug("TTS Synthesizer: Received EOS token, waiting for pending jobs...")

                        # Wait for all pending futures to complete
                        for seq_num in sorted(self.pending_futures.keys()):
                            future = self.pending_futures[seq_num]
                            try:
                                audio_msg = future.result(timeout=60)  # Max 60s per job
                                self.audio_output_queue.put(audio_msg)
                                logger.debug(f"TTS Synthesizer: Sent pending job #{seq_num} to audio queue")
                            except Exception as e:
                                logger.error(f"TTS Synthesizer: Failed to get result for job #{seq_num}: {e}")

                        self.pending_futures.clear()

                        # Send EOS to audio queue
                        self.audio_output_queue.put(
                            AudioMessage(
                                audio=np.array([], dtype=np.float32),
                                text="",
                                is_eos=True,
                                sequence_num=0
                            )
                        )
                        continue

                    # Skip empty text
                    if not text_msg.text or not text_msg.text.strip():
                        logger.warning(f"TTS Synthesizer: Received empty text message #{text_msg.sequence_num}")
                        continue

                    logger.info(f"TTS Synthesizer: Queuing job #{text_msg.sequence_num}: '{text_msg.text[:50]}...'")

                    # Submit to thread pool for parallel processing
                    future = self.executor.submit(self._process_text, text_msg)
                    self.pending_futures[text_msg.sequence_num] = future

                    # Check for completed futures and send in order
                    self._send_completed_results()

                except queue.Empty:
                    # Check for completed futures even when queue is empty
                    self._send_completed_results()
                    continue

                except Exception as e:
                    logger.exception(f"TextToSpeechSynthesizer: Unexpected error in run loop: {e}")
                    time.sleep(self.pause_time)

        finally:
            # Shutdown thread pool
            logger.info("TTS Synthesizer: Shutting down thread pool...")
            self.executor.shutdown(wait=True, cancel_futures=True)
            logger.info("TextToSpeechSynthesizer thread finished.")

    def _send_completed_results(self) -> None:
        """
        Check pending futures and send completed results to audio queue in sequence order.
        Only sends results if they can be sent in the correct order.
        """
        # Find the lowest pending sequence number
        if not self.pending_futures:
            return

        min_seq = min(self.pending_futures.keys())

        # Send all completed results in order starting from min_seq
        while min_seq in self.pending_futures:
            future = self.pending_futures[min_seq]

            # Only process if done
            if not future.done():
                break

            # Get result and send to audio queue
            try:
                audio_msg = future.result(timeout=0)  # Should be instant since done()
                self.audio_output_queue.put(audio_msg)
                logger.debug(f"TTS Synthesizer: Sent job #{min_seq} to audio queue")
            except Exception as e:
                logger.error(f"TTS Synthesizer: Failed to get result for job #{min_seq}: {e}")
                # Send empty audio on error
                self.audio_output_queue.put(
                    AudioMessage(
                        audio=np.array([], dtype=np.float32),
                        text="",
                        is_eos=False,
                        sequence_num=min_seq
                    )
                )

            # Remove from pending
            del self.pending_futures[min_seq]

            # Move to next sequence number
            if self.pending_futures:
                min_seq = min(self.pending_futures.keys())
            else:
                break
