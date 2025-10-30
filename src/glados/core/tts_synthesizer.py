import queue
import threading

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

    def run(self) -> None:
        """
        Starts the main loop for the TTS Synthesizer thread with sequential processing.

        This method continuously checks the TTS input queue for text to synthesize.
        It processes text messages sequentially and sends results to the audio output queue.

        The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("TTS Synthesizer started (sequential mode)")

        while not self.shutdown_event.is_set():
            try:
                text_msg = self.tts_input_queue.get(timeout=self.pause_time)

                if text_msg.is_eos:
                    self.audio_output_queue.put(AudioMessage(
                        audio=np.array([], dtype=np.float32),
                        text="", is_eos=True, sequence_num=0
                    ))
                    continue

                if not text_msg.text or not text_msg.text.strip():
                    continue

                # Sequential processing
                try:
                    spoken_text = self.stc.text_to_spoken(text_msg.text)
                    audio_data = self.tts_model.generate_speech_audio(spoken_text)

                    self.audio_output_queue.put(AudioMessage(
                        audio=audio_data,
                        text=spoken_text,
                        is_eos=False,
                        sequence_num=text_msg.sequence_num
                    ))
                    logger.info(f"TTS completed: '{spoken_text[:50]}...'")

                except Exception as e:
                    logger.error(f"TTS failed: {e}")

            except queue.Empty:
                continue

        logger.info("TTS Synthesizer finished")
