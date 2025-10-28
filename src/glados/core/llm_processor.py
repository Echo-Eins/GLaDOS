# --- llm_processor.py ---
import json
import math
import queue
import re
import threading
import time
from typing import Any, ClassVar

from .audio_data import RecognitionResult, TTSTextMessage

from loguru import logger
from pydantic import HttpUrl  # If HttpUrl is used by config
import requests


def sanitize_emotions_for_json(emotions: dict[str, float] | None) -> dict[str, float] | None:
    """Sanitize emotion values to ensure JSON compatibility.

    Replaces NaN and infinity values with 0.0 to prevent JSON serialization errors.

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
        logger.warning(
            f"LLM Processor: Sanitized invalid emotion values before JSON serialization. "
            f"Original: {emotions}, Sanitized: {sanitized}"
        )

    return sanitized


class LanguageModelProcessor:
    """
    A thread that processes text input for a language model, streaming responses and sending them to TTS.
    This class is designed to run in a separate thread, continuously checking for new text to process
    until a shutdown event is set. It handles conversation history, manages streaming responses,
    and sends synthesized sentences to a TTS queue.
    """

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n", "\n\n"}
    PARAGRAPH_BATCH_SIZE: ClassVar[int] = 3  # Process 3 paragraphs in parallel

    def __init__(
        self,
        llm_input_queue: queue.Queue[RecognitionResult],
        tts_input_queue: queue.Queue[TTSTextMessage],
        conversation_history: list[dict[str, str]],  # Shared
        completion_url: HttpUrl,
        model_name: str,  # Renamed from 'model' to avoid conflict
        api_key: str | None,
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
    ) -> None:
        self.llm_input_queue = llm_input_queue
        self.tts_input_queue = tts_input_queue
        self.conversation_history = conversation_history
        self.completion_url = completion_url
        self.model_name = model_name
        self.api_key = api_key
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

        self.prompt_headers = {"Content-Type": "application/json"}
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"

        # Sequence counter for paragraph ordering
        self.sequence_counter = 0

    def _clean_raw_bytes(self, line: bytes) -> dict[str, str] | None:
        """
        Clean and parse a raw byte line from the LLM response.
        Handles both OpenAI and Ollama formats, returning a dictionary or None if parsing fails.

        Args:
            line (bytes): The raw byte line from the LLM response.
        Returns:
            dict[str, str] | None: Parsed JSON dictionary or None if parsing fails.
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]
                if json_str.strip() == "[DONE]":  # Handle OpenAI [DONE] marker
                    return {"done_marker": "True"}
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except json.JSONDecodeError:
            # If it's not JSON, it might be Ollama's final summary object which isn't part of the stream
            # Or just noise.
            logger.trace(
                f"LLM Processor: Failed to parse non-JSON server response line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )  # Log only a part
            return None
        except Exception as e:
            logger.warning(
                f"LLM Processor: Failed to parse server response: {e} for line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        # Copy from Glados._process_chunk
        if not line or not isinstance(line, dict):
            return None
        try:
            # Handle OpenAI format
            if line.get("done_marker"):  # Handle [DONE] marker
                return None
            elif "choices" in line:  # OpenAI format
                content = line.get("choices", [{}])[0].get("delta", {}).get("content")
                return str(content) if content else None
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                return content if content else None
        except Exception as e:
            logger.error(f"LLM Processor: Error processing chunk: {e}, chunk: {line}")
            return None

    def _send_paragraph_batch(self, paragraphs: list[str]) -> None:
        """
        Send a batch of paragraphs to TTS queue with sequence numbers.
        Batch will be processed in parallel by TTS synthesizer.

        Args:
            paragraphs: List of paragraph strings to send (up to PARAGRAPH_BATCH_SIZE)
        """
        if not paragraphs:
            return

        logger.info(f"LLM Processor: Sending batch of {len(paragraphs)} paragraphs to TTS")

        for paragraph in paragraphs:
            # Clean paragraph for TTS
            # Transform square brackets: [ПРОТОКОЛ] Текст → ПРОТОКОЛ: Текст
            # This preserves visual formatting while creating natural TTS intonation
            cleaned = re.sub(r"\[([^\]]+)\]", r"\1:", paragraph)
            # Remove markdown formatting (asterisks and round brackets with content)
            cleaned = re.sub(r"\*.*?\*|\(.*?\)", "", cleaned)
            cleaned = cleaned.replace("  ", " ").strip()

            if cleaned and len(cleaned) > 1:  # Avoid empty or single-char strings
                msg = TTSTextMessage(
                    text=cleaned,
                    sequence_num=self.sequence_counter,
                    is_eos=False
                )
                self.sequence_counter += 1
                logger.debug(f"LLM Processor: Queuing paragraph #{msg.sequence_num}: '{cleaned[:50]}...'")
                self.tts_input_queue.put(msg)

    def run(self) -> None:
        """
        Starts the main loop for the LanguageModelProcessor thread.

        This method continuously checks the LLM input queue for text to process.
        It processes the text, sends it to the LLM API, and streams the response.
        It handles conversation history, manages streaming responses, and sends synthesized sentences
        to a TTS queue. The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("LanguageModelProcessor thread started.")
        while not self.shutdown_event.is_set():
            try:
                recognition = self.llm_input_queue.get(timeout=self.pause_time)
                detected_text = recognition.text
                emotions = recognition.emotions

                if not detected_text:
                    logger.info("LLM Processor: Received empty ASR result, skipping request.")
                    continue

                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
                    # Ensure EOS is sent if a previous stream was cut short by this interruption
                    # This logic might need refinement based on state. For now, assume no prior stream.
                    continue

                logger.info(f"LLM Processor: Received text for LLM: '{detected_text}'")

                if emotions:
                    logger.info(f"LLM Processor: Emotion probabilities {dict(emotions)}")

                self.conversation_history.append({"role": "user", "content": detected_text})

                data = {
                    "model": self.model_name,
                    "stream": True,
                    "messages": self.conversation_history,
                    # Add other parameters like temperature, max_tokens if needed from config
                }

                if emotions:
                    # Sanitize emotions to prevent JSON serialization errors with NaN/inf values
                    sanitized_emotions = sanitize_emotions_for_json(emotions)
                    if sanitized_emotions:
                        data["metadata"] = {"emotions": sanitized_emotions}

                # Paragraph-based parsing for parallel TTS processing
                paragraph_buffer: list[str] = []  # Current paragraph being built
                paragraph_batch: list[str] = []  # Batch of complete paragraphs (max 3)

                try:
                    with requests.post(
                        str(self.completion_url),
                        headers=self.prompt_headers,
                        json=data,
                        stream=True,
                        timeout=30,  # Add a timeout for the request itself
                    ) as response:
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        logger.debug("LLM Processor: Request to LLM successful, processing stream...")

                        for line in response.iter_lines():
                            if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                                logger.info("LLM Processor: Interruption or shutdown detected during LLM stream.")
                                break  # Stop processing stream

                            if line:
                                cleaned_line_data = self._clean_raw_bytes(line)
                                if cleaned_line_data:
                                    chunk = self._process_chunk(cleaned_line_data)
                                    if chunk:  # Chunk can be an empty string, but None means no actual content
                                        paragraph_buffer.append(chunk)

                                        # Check for paragraph break (double newline)
                                        if "\n\n" in chunk:
                                            # Complete paragraph detected
                                            paragraph_text = "".join(paragraph_buffer)
                                            paragraph_buffer = []

                                            # Add to batch
                                            paragraph_batch.append(paragraph_text)

                                            # Send batch when we have 3 paragraphs
                                            if len(paragraph_batch) >= self.PARAGRAPH_BATCH_SIZE:
                                                self._send_paragraph_batch(paragraph_batch)
                                                paragraph_batch = []

                                        # Also split on sentence endings if no paragraph break for a while
                                        elif chunk.strip() in {".", "!", "?", "?!"} and len("".join(paragraph_buffer)) > 200:
                                            # Long paragraph without break - split it
                                            paragraph_text = "".join(paragraph_buffer)
                                            paragraph_buffer = []
                                            paragraph_batch.append(paragraph_text)

                                            if len(paragraph_batch) >= self.PARAGRAPH_BATCH_SIZE:
                                                self._send_paragraph_batch(paragraph_batch)
                                                paragraph_batch = []

                                    # OpenAI [DONE]
                                    elif cleaned_line_data.get("done_marker"):  # OpenAI [DONE]
                                        break
                                    # Ollama end
                                    elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                                        break

                        # After loop, process any remaining content if not interrupted
                        if self.processing_active_event.is_set():
                            # Add remaining paragraph buffer to batch
                            if paragraph_buffer:
                                paragraph_text = "".join(paragraph_buffer)
                                paragraph_batch.append(paragraph_text)

                            # Send any remaining paragraphs
                            if paragraph_batch:
                                self._send_paragraph_batch(paragraph_batch)

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"LLM Processor: Connection error to LLM service: {e}")
                    error_msg = TTSTextMessage(
                        text="I'm unable to connect to my thinking module. Please check the LLM service connection.",
                        sequence_num=self.sequence_counter,
                        is_eos=False
                    )
                    self.sequence_counter += 1
                    self.tts_input_queue.put(error_msg)
                except requests.exceptions.Timeout as e:
                    logger.error(f"LLM Processor: Request to LLM timed out: {e}")
                    error_msg = TTSTextMessage(
                        text="My brain seems to be taking too long to respond. It might be overloaded.",
                        sequence_num=self.sequence_counter,
                        is_eos=False
                    )
                    self.sequence_counter += 1
                    self.tts_input_queue.put(error_msg)
                except requests.exceptions.HTTPError as e:
                    status_code = (
                        e.response.status_code
                        if hasattr(e, "response") and hasattr(e.response, "status_code")
                        else "unknown"
                    )
                    logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                    error_msg = TTSTextMessage(
                        text=f"I received an error from my thinking module. HTTP status {status_code}.",
                        sequence_num=self.sequence_counter,
                        is_eos=False
                    )
                    self.sequence_counter += 1
                    self.tts_input_queue.put(error_msg)
                except requests.exceptions.RequestException as e:
                    logger.error(f"LLM Processor: Request to LLM failed: {e}")
                    error_msg = TTSTextMessage(
                        text="Sorry, I encountered an error trying to reach my brain.",
                        sequence_num=self.sequence_counter,
                        is_eos=False
                    )
                    self.sequence_counter += 1
                    self.tts_input_queue.put(error_msg)
                except Exception as e:
                    logger.exception(f"LLM Processor: Unexpected error during LLM request/streaming: {e}")
                    error_msg = TTSTextMessage(
                        text="I'm having a little trouble thinking right now.",
                        sequence_num=self.sequence_counter,
                        is_eos=False
                    )
                    self.sequence_counter += 1
                    self.tts_input_queue.put(error_msg)
                finally:
                    # Always send EOS if we started processing, unless interrupted early
                    if self.processing_active_event.is_set():  # Only send EOS if not interrupted
                        logger.debug("LLM Processor: Sending EOS token to TTS queue.")
                        eos_msg = TTSTextMessage(text="", sequence_num=0, is_eos=True)
                        self.tts_input_queue.put(eos_msg)
                    else:
                        logger.info("LLM Processor: Interrupted, not sending EOS from LLM processing.")
                        # The AudioPlayer will handle clearing its state.
                        # If an EOS was already sent by TTS from a *previous* partial sentence,
                        # this could lead to an early clear of currently_speaking.
                        # The `processing_active_event` is key to synchronize.

            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"LLM Processor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("LanguageModelProcessor thread finished.")
