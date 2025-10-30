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
        keep_alive_timeout: str = "30m",  # How long to keep model loaded in Ollama
        enable_thinking: bool = False,  # Enable thinking mode by default
        thinking_trigger_words: list[str] | None = None,  # Keywords that activate thinking mode
        thinking_fuzzy_threshold: float = 0.75,  # Fuzzy matching threshold
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
        self.keep_alive_timeout = keep_alive_timeout
        self.enable_thinking = enable_thinking
        self.thinking_trigger_words = thinking_trigger_words or []
        self.thinking_fuzzy_threshold = thinking_fuzzy_threshold

        self.prompt_headers = {"Content-Type": "application/json"}
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"

        # Sequence counter for paragraph ordering
        self.sequence_counter = 0

        # Paragraph buffer for accumulating text until \n\n
        self.paragraph_buffer_text = ""

        # Chat logger (will be set by engine after initialization)
        self.chat_logger = None

    def _should_enable_thinking(self, user_text: str) -> bool:
        """
        Determine if thinking mode should be enabled for the given user text.
        Uses fuzzy matching to check if any trigger words are present.

        Args:
            user_text: The user's input text

        Returns:
            bool: True if thinking mode should be enabled
        """
        if not self.thinking_trigger_words:
            return self.enable_thinking  # Return default if no trigger words configured

        user_text_lower = user_text.lower()

        # Try using rapidfuzz for better performance (fallback to simple matching if not available)
        try:
            from rapidfuzz import fuzz

            for keyword in self.thinking_trigger_words:
                # Use partial_ratio for substring matching
                ratio = fuzz.partial_ratio(keyword.lower(), user_text_lower)
                if ratio >= self.thinking_fuzzy_threshold * 100:
                    logger.info(f"Thinking mode activated by keyword '{keyword}' (match: {ratio}%)")
                    return True
        except ImportError:
            # Fallback to simple substring matching if rapidfuzz not available
            logger.debug("rapidfuzz not available, using simple substring matching")
            for keyword in self.thinking_trigger_words:
                if keyword.lower() in user_text_lower:
                    logger.info(f"Thinking mode activated by keyword '{keyword}'")
                    return True

        return self.enable_thinking  # Return default if no matches

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

                # Log TTS paragraph to chat logger
                if self.chat_logger:
                    self.chat_logger.log_tts_paragraph(cleaned, msg.sequence_num)

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

                # Check for empty or whitespace-only text
                if not detected_text or not detected_text.strip():
                    logger.info("LLM Processor: Received empty ASR result (whitespace only), skipping request.")
                    continue

                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
                    # Ensure EOS is sent if a previous stream was cut short by this interruption
                    # This logic might need refinement based on state. For now, assume no prior stream.
                    continue

                logger.info(f"LLM Processor: Received text for LLM: '{detected_text}'")

                # Format user message with emotions if available and valid
                user_content = detected_text
                if emotions:
                    logger.info(f"LLM Processor: Emotion probabilities {dict(emotions)}")
                    # Sanitize emotions to prevent JSON serialization errors with NaN/inf values
                    sanitized_emotions = sanitize_emotions_for_json(emotions)

                    # Only append emotions if at least one value is meaningful (> 0.01)
                    if sanitized_emotions and any(val > 0.01 for val in sanitized_emotions.values()):
                        # Append emotions to user message in format: "text" {'emotion': value, ...}
                        user_content = f'"{detected_text}" {sanitized_emotions}'
                        logger.debug(f"LLM Processor: User message with emotions: {user_content}")
                    else:
                        logger.debug("LLM Processor: All emotions near zero, not appending to message")

                self.conversation_history.append({"role": "user", "content": user_content})

                # Log user message to chat logger
                if self.chat_logger:
                    self.chat_logger.log_message("user", user_content)

                # Determine if thinking mode should be enabled for this request
                enable_thinking_for_request = self._should_enable_thinking(detected_text)

                data = {
                    "model": self.model_name,
                    "stream": True,
                    "messages": self.conversation_history,
                    "keep_alive": self.keep_alive_timeout,  # Keep model loaded in Ollama memory
                }

                # Add thinking mode configuration for Ollama (GLM-4.6 via vLLM/SGLang format)
                if self.thinking_trigger_words:  # Only add if trigger words are configured
                    data["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": enable_thinking_for_request,
                            "output_thinking": False  # CRITICAL: Disable thinking blocks in output
                        }
                    }
                    if enable_thinking_for_request:
                        logger.info("Thinking mode enabled for this request (thinking blocks hidden from output)")
                    else:
                        logger.debug("Thinking mode disabled for this request")

                # Clear paragraph buffer for new request
                self.paragraph_buffer_text = ""

                # Paragraph-based parsing for parallel TTS processing
                paragraph_batch: list[str] = []  # Batch of complete paragraphs

                # Accumulate full assistant response for logging
                assistant_response_chunks: list[str] = []

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
                                        # Accumulate chunk for full response logging
                                        assistant_response_chunks.append(chunk)

                                        # Add chunk to running buffer
                                        self.paragraph_buffer_text += chunk

                                        # Check for paragraph breaks in accumulated buffer
                                        while "\n\n" in self.paragraph_buffer_text:
                                            # Find first paragraph break
                                            idx = self.paragraph_buffer_text.find("\n\n")
                                            paragraph_text = self.paragraph_buffer_text[:idx]
                                            # Keep remaining text after \n\n for next paragraph
                                            self.paragraph_buffer_text = self.paragraph_buffer_text[idx+2:]

                                            # Add completed paragraph to batch
                                            if paragraph_text.strip():  # Skip empty paragraphs
                                                paragraph_batch.append(paragraph_text)

                                                # Send batch immediately when we have a complete paragraph
                                                # (dynamic batching - no need to wait for 3 paragraphs)
                                                if len(paragraph_batch) >= 1:
                                                    self._send_paragraph_batch(paragraph_batch)
                                                    paragraph_batch = []

                                        # Fallback: split long paragraphs without breaks (>500 chars)
                                        # This prevents buffering indefinitely for very long text without \n\n
                                        if len(self.paragraph_buffer_text) > 500:
                                            # Look for sentence ending
                                            last_period = max(
                                                self.paragraph_buffer_text.rfind('.'),
                                                self.paragraph_buffer_text.rfind('!'),
                                                self.paragraph_buffer_text.rfind('?')
                                            )
                                            if last_period > 200:  # Ensure meaningful chunk
                                                paragraph_text = self.paragraph_buffer_text[:last_period+1]
                                                self.paragraph_buffer_text = self.paragraph_buffer_text[last_period+1:]

                                                paragraph_batch.append(paragraph_text)
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
                            # Add remaining buffer to batch
                            if self.paragraph_buffer_text.strip():
                                paragraph_batch.append(self.paragraph_buffer_text)
                                self.paragraph_buffer_text = ""  # Clear buffer

                            # Send any remaining paragraphs
                            if paragraph_batch:
                                self._send_paragraph_batch(paragraph_batch)

                        # Log full assistant response to conversation history and chat logger
                        full_assistant_response = "".join(assistant_response_chunks)
                        if full_assistant_response.strip():
                            self.conversation_history.append({"role": "assistant", "content": full_assistant_response})
                            if self.chat_logger:
                                self.chat_logger.log_message("assistant", full_assistant_response)

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
