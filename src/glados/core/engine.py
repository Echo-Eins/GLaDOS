"""
Core engine module for the Glados voice assistant.

This module provides the main orchestration classes including the Glados assistant,
configuration management, and component coordination.
"""

from pathlib import Path
import queue
import sys
import threading
import time

from loguru import logger
from pydantic import BaseModel, HttpUrl
import yaml

import numpy as np
from numpy.typing import NDArray

from ..ASR import TranscriberProtocol, get_audio_transcriber
from ..audio_io import AudioProtocol, get_audio_system
from ..TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from .audio_data import AudioMessage, RecognitionResult, TTSTextMessage
from .chat_logger import ChatLogger
from .llm_processor import LanguageModelProcessor
from .speech_listener import SpeechListener
from .speech_player import SpeechPlayer
from .tts_synthesizer import TextToSpeechSynthesizer

logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")


class PersonalityPrompt(BaseModel):
    """
    Represents a single personality prompt message for the assistant.

    Contains exactly one of: system, user, or assistant message content.
    Used to configure the assistant's personality and behavior.
    """

    system: str | None = None
    user: str | None = None
    assistant: str | None = None

    def to_chat_message(self) -> dict[str, str]:
        """Convert the prompt to a chat message format.

        Returns:
            dict[str, str]: A single chat message dictionary

        Raises:
            ValueError: If the prompt does not contain exactly one non-null field
        """
        fields = self.model_dump(exclude_none=True)
        if len(fields) != 1:
            raise ValueError("PersonalityPrompt must have exactly one non-null field")

        field, value = next(iter(fields.items()))
        return {"role": field, "content": value}


class GladosConfig(BaseModel):
    """
    Configuration model for the Glados voice assistant.

    Defines all necessary parameters for initializing the assistant including
    LLM settings, audio I/O backend, ASR/TTS engines, and personality configuration.
    Supports loading from YAML files with nested key navigation.
    """

    llm_model: str
    completion_url: HttpUrl
    api_key: str | None
    keep_alive_timeout: str = "30m"  # How long to keep LLM loaded in Ollama (e.g., "5m", "30m", "1h")
    enable_thinking: bool = False  # Enable thinking mode by default (auto-enabled for complex queries)
    thinking_trigger_words: list[str] = []  # Keywords that activate thinking mode via fuzzy matching
    thinking_fuzzy_threshold: float = 0.75  # Fuzzy matching threshold (0.0-1.0)
    interruptible: bool
    audio_io: str
    asr_engine: str
    language: str = "en"
    wake_word: str | None
    voice: str
    announcement: str | None
    personality_preprompt: list[PersonalityPrompt]

    # EN-Branch Integration (optional fields for bilingual mode)
    enable_en_branch: bool = False  # Enable parallel EN pipeline
    asr_ru_engine: str | None = None  # Russian ASR engine
    asr_en_engine: str | None = None  # English ASR engine
    voice_ru: str | None = None  # Russian TTS voice
    voice_en: str | None = None  # English TTS voice
    language_detection: dict | None = None  # Language detection settings
    audio_mixer: dict | None = None  # Audio mixer settings

    @classmethod
    def from_yaml(cls, path: str | Path, key_to_config: tuple[str, ...] = ("Glados",)) -> "GladosConfig":
        """
        Load a GladosConfig instance from a YAML configuration file.

        Parameters:
            path: Path to the YAML configuration file
            key_to_config: Tuple of keys to navigate nested configuration

        Returns:
            GladosConfig: Configuration object with validated settings

        Raises:
            ValueError: If the YAML content is invalid
            OSError: If the file cannot be read
            pydantic.ValidationError: If the configuration is invalid
        """
        path = Path(path)

        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise ValueError(f"Could not decode YAML file {path} with any supported encoding")

        # Navigate through nested keys
        config = data
        for key in key_to_config:
            config = config[key]

        return cls.model_validate(config)

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]


class Glados:
    """
    Glados voice assistant orchestrator.
    This class manages the components of the Glados voice assistant, including speech recognition,
    language model processing, text-to-speech synthesis, and audio playback.
    It initializes the necessary components, starts background threads for processing, and provides
    methods for interaction with the assistant.
    """

    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
    NEUROTOXIN_RELEASE_ALLOWED: bool = False  # preparation for function calling, see issue #13
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: TranscriberProtocol,
        tts_model: SpeechSynthesizerProtocol,
        audio_io: AudioProtocol,
        completion_url: HttpUrl,
        llm_model: str,
        api_key: str | None = None,
        keep_alive_timeout: str = "30m",
        enable_thinking: bool = False,
        thinking_trigger_words: list[str] | None = None,
        thinking_fuzzy_threshold: float = 0.75,
        interruptible: bool = True,
        wake_word: str | None = None,
        announcement: str | None = None,
        personality_preprompt: tuple[dict[str, str], ...] = DEFAULT_PERSONALITY_PREPROMPT,
        # EN-Branch parameters (bilingual mode)
        enable_en_branch: bool = False,
        ru_asr_model: TranscriberProtocol | None = None,
        en_asr_model: TranscriberProtocol | None = None,
        ru_tts_model: SpeechSynthesizerProtocol | None = None,
        en_tts_model: SpeechSynthesizerProtocol | None = None,
        lid_model: "object | None" = None,  # SileroLanguageID
        language_detection_config: dict | None = None,
        audio_mixer_config: dict | None = None,
    ) -> None:
        """
        Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Supports both monolingual (legacy) and bilingual (EN-Branch) modes.

        Args:
            asr_model (TranscriberProtocol): The ASR model for transcribing audio input (legacy/main).
            tts_model (SpeechSynthesizerProtocol): The TTS model for synthesizing spoken output (legacy/main).
            audio_io (AudioProtocol): The audio input/output system to use.
            completion_url (HttpUrl): The URL for the LLM completion endpoint.
            llm_model (str): The name of the LLM model to use.
            api_key (str | None): API key for accessing the LLM service, if required.
            keep_alive_timeout (str): How long to keep LLM loaded in Ollama (e.g., "5m", "30m", "1h").
            enable_thinking (bool): Enable thinking mode by default for all requests.
            thinking_trigger_words (list[str] | None): Keywords that activate thinking mode via fuzzy matching.
            thinking_fuzzy_threshold (float): Fuzzy matching threshold (0.0-1.0) for trigger words.
            interruptible (bool): Whether the assistant can be interrupted while speaking.
            wake_word (str | None): Optional wake word to trigger the assistant.
            announcement (str | None): Optional announcement to play on startup.
            personality_preprompt (tuple[dict[str, str], ...]): Initial personality preprompt messages.
            enable_en_branch (bool): Enable parallel EN pipeline for bilingual mode.
            ru_asr_model (TranscriberProtocol | None): Russian ASR model (bilingual mode).
            en_asr_model (TranscriberProtocol | None): English ASR model (bilingual mode).
            ru_tts_model (SpeechSynthesizerProtocol | None): Russian TTS model (bilingual mode).
            en_tts_model (SpeechSynthesizerProtocol | None): English TTS model (bilingual mode).
            lid_model (object | None): Language ID model (Silero LID).
            language_detection_config (dict | None): Language detection configuration.
            audio_mixer_config (dict | None): Audio mixer configuration.
        """
        self._asr_model = asr_model
        self._tts = tts_model
        self.completion_url = completion_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.keep_alive_timeout = keep_alive_timeout
        self.enable_thinking = enable_thinking
        self.thinking_trigger_words = thinking_trigger_words or []
        self.thinking_fuzzy_threshold = thinking_fuzzy_threshold
        self.interruptible = interruptible
        self.wake_word = wake_word
        self.announcement = announcement
        self._messages: list[dict[str, str]] = list(personality_preprompt)

        # Spectrum sharing infrastructure for UI widgets
        self._spectrum_band_count = 16
        self.spectrum_queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=256)
        self._register_spectrum_stream()

        # Initialize spoken text converter, that converts text to spoken text. eg. 12 -> "twelve"
        self._stc = stc.SpokenTextConverter()

        # Initialize events for thread synchronization
        self.processing_active_event = threading.Event()  # Indicates if input processing is active (ASR + LLM + TTS)
        self.currently_speaking_event = threading.Event()  # Indicates if the assistant is currently speaking
        self.shutdown_event = threading.Event()  # Event to signal shutdown of all threads
        self.pipeline_status_queue: queue.Queue[tuple[str, bool]] = queue.Queue(maxsize=32)

        # Initialize queues for inter-thread communication
        self.llm_queue: queue.Queue[RecognitionResult] = queue.Queue()
        self.tts_queue: queue.Queue[TTSTextMessage] = queue.Queue()  # TTSTextMessage from LLMProcessor to TTSynthesizer
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()  # AudioMessages from TTSSynthesizer to AudioPlayer

        # Initialize audio input/output system
        self.audio_io: AudioProtocol = audio_io
        logger.info("Audio input started successfully.")

        # Initialize threads for each component
        self.component_threads: list[threading.Thread] = []

        self.speech_listener = SpeechListener(
            audio_io=self.audio_io,
            llm_queue=self.llm_queue,
            asr_model=self._asr_model,
            wake_word=self.wake_word,
            interruptible=self.interruptible,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        self.llm_processor = LanguageModelProcessor(
            llm_input_queue=self.llm_queue,
            tts_input_queue=self.tts_queue,
            conversation_history=self._messages,  # Shared, to be refactored
            completion_url=self.completion_url,
            model_name=self.llm_model,
            api_key=self.api_key,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
            keep_alive_timeout=self.keep_alive_timeout,
            enable_thinking=self.enable_thinking,
            thinking_trigger_words=self.thinking_trigger_words,
            thinking_fuzzy_threshold=self.thinking_fuzzy_threshold,
            status_queue=self.pipeline_status_queue,
        )

        self.tts_synthesizer = TextToSpeechSynthesizer(
            tts_input_queue=self.tts_queue,
            audio_output_queue=self.audio_queue,
            tts_model=self._tts,
            stc_instance=self._stc,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
        )

        self.speech_player = SpeechPlayer(
            audio_io=self.audio_io,
            audio_output_queue=self.audio_queue,
            conversation_history=self._messages,  # Shared, to be refactored
            tts_sample_rate=self._tts.sample_rate,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        # Initialize EN-Branch components (bilingual mode)
        self.enable_en_branch = enable_en_branch
        self.language_router = None
        self.ru_branch_processor = None
        self.en_branch_processor = None
        self.audio_mixer = None

        if self.enable_en_branch:
            logger.info("EN-Branch: Initializing bilingual mode components...")

            # Validate required models
            if not all([ru_asr_model, en_asr_model, ru_tts_model, en_tts_model, lid_model]):
                logger.warning(
                    "EN-Branch: Missing required models for bilingual mode. "
                    "Falling back to monolingual mode."
                )
                self.enable_en_branch = False
                self.speech_listener.set_language_router(None)
            else:
                try:
                    # Import EN-Branch modules
                    from .language_router import LanguageRouter
                    from .branch_processor import create_branch_processors
                    from .audio_mixer import AudioMixer

                    # Create language-specific queues
                    self.ru_segments_queue: queue.Queue = queue.Queue()
                    self.en_segments_queue: queue.Queue = queue.Queue()
                    self.mixer_output_queue: queue.Queue = queue.Queue()

                    # Get language detection config
                    lid_config = language_detection_config or {}
                    confidence_threshold = lid_config.get("confidence_threshold", 0.7)
                    default_language = lid_config.get("default_language", "ru")

                    # Create Language Router
                    logger.info("EN-Branch: Creating Language Router...")
                    self.language_router = LanguageRouter(
                        lid_model=lid_model,
                        ru_queue=self.ru_segments_queue,
                        en_queue=self.en_segments_queue,
                        confidence_threshold=confidence_threshold,
                        default_language=default_language,
                        shutdown_event=self.shutdown_event,
                    )

                    # Connect speech listener to bilingual router so it can dispatch audio segments
                    self.speech_listener.set_language_router(self.language_router)

                    # Create Branch Processors
                    logger.info("EN-Branch: Creating RU and EN Branch Processors...")
                    self.ru_branch_processor, self.en_branch_processor = create_branch_processors(
                        ru_input_queue=self.ru_segments_queue,
                        en_input_queue=self.en_segments_queue,
                        output_queue=self.mixer_output_queue,
                        ru_asr_model=ru_asr_model,
                        en_asr_model=en_asr_model,
                        ru_tts_model=ru_tts_model,
                        en_tts_model=en_tts_model,
                        stc_instance=self._stc,
                        shutdown_event=self.shutdown_event,
                        pause_time=self.PAUSE_TIME,
                    )

                    # Create Audio Mixer
                    logger.info("EN-Branch: Creating Audio Mixer...")
                    mixer_config = audio_mixer_config or {}
                    target_sample_rate = mixer_config.get("target_sample_rate", 48000)

                    self.audio_mixer = AudioMixer(
                        input_queue=self.mixer_output_queue,
                        output_queue=self.audio_queue,
                        target_sample_rate=target_sample_rate,
                        shutdown_event=self.shutdown_event,
                        pause_time=self.PAUSE_TIME,
                    )

                    logger.success("EN-Branch: Bilingual mode components initialized successfully!")

                except Exception as e:
                    logger.error(f"EN-Branch: Failed to initialize bilingual components: {e}")
                    logger.exception(e)
                    logger.warning("EN-Branch: Falling back to monolingual mode.")
                    self.enable_en_branch = False
                    self.language_router = None
                    self.speech_listener.set_language_router(None)

        # Build thread targets dictionary
        thread_targets = {
            "SpeechListener": self.speech_listener.run,
            "LLMProcessor": self.llm_processor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.speech_player.run,
        }

        # Add EN-Branch threads if enabled
        if self.enable_en_branch and self.ru_branch_processor and self.en_branch_processor and self.audio_mixer:
            thread_targets.update({
                "RU_BranchProcessor": self.ru_branch_processor.run,
                "EN_BranchProcessor": self.en_branch_processor.run,
                "AudioMixer": self.audio_mixer.run,
            })
            logger.info("EN-Branch: Added bilingual processing threads to orchestrator.")

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")

        # Initialize chat logger and start session
        self.chat_logger = ChatLogger()
        self.chat_logger.start_session(self._messages)
        # Pass chat logger to LLM processor for message logging
        self.llm_processor.chat_logger = self.chat_logger

        # warm up LLM model to pre-load it into Ollama memory in background thread
        # IMPORTANT: Do this AFTER all threads are started so audio system is operational
        # This allows user to interact with GLaDOS while warmup happens in background
        self._start_llm_warmup_thread()

    def _register_spectrum_stream(self) -> None:
        """Connect the TTS pipeline spectrum output to an internal queue for the UI."""

        register = getattr(self._tts, "register_spectrum_consumer", None)
        if not callable(register):
            return

        def _consumer(bands: NDArray[np.float32]) -> None:
            try:
                frame = np.asarray(bands, dtype=np.float32).copy()
            except Exception:
                frame = np.array(bands, dtype=np.float32, copy=True)

            try:
                self.spectrum_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.spectrum_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.spectrum_queue.put_nowait(frame)
                except queue.Full:
                    pass

        try:
            register(_consumer, band_count=self._spectrum_band_count)
        except TypeError:
            register(_consumer)

    def _start_llm_warmup_thread(self) -> None:
        """
        Start LLM warmup in a background thread so it doesn't block initialization.

        This allows the audio system and all other components to become operational immediately,
        while the LLM model loads in the background. The first user request may still experience
        a delay if warmup hasn't completed yet, but the system remains responsive.
        """
        def warmup_wrapper():
            logger.info("LLM Warmup: Starting background warmup (this may take 1-2 minutes)...")
            logger.info("LLM Warmup: Audio system is operational - you can start speaking!")
            self._warmup_llm()

        warmup_thread = threading.Thread(
            target=warmup_wrapper,
            name="LLMWarmup",
            daemon=True  # Don't block shutdown
        )
        warmup_thread.start()
        # Don't join - let it run in background

    def _warmup_llm(self) -> None:
        """
        Pre-load LLM model into Ollama memory AND process personality preprompt.

        This method sends the full personality preprompt to the LLM to ensure that:
        1. The model is loaded into VRAM
        2. The large system prompt is processed and cached
        3. First user request will be instant (no prompt processing delay)

        The warmup response is consumed and discarded - it will never reach TTS queues
        since this happens BEFORE thread initialization.

        If the warmup fails, it logs a warning but does not prevent system startup.
        """
        import requests

        logger.info(f"LLM Warmup: Pre-loading model '{self.llm_model}' and processing system prompt...")

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Use /api/chat with personality preprompt to warm up the full context
            # This ensures the large system prompt is processed during startup
            warmup_messages = list(self._messages)  # Copy personality preprompt

            # Add an empty user message to trigger a minimal response
            # The LLM is instructed in the system prompt not to respond to empty messages,
            # but even if it does, we consume and discard the response safely here.
            warmup_messages.append({"role": "user", "content": ""})

            payload = {
                "model": self.llm_model,
                "messages": warmup_messages,
                "stream": False,  # IMPORTANT: non-streaming to get complete response at once
                "keep_alive": self.keep_alive_timeout,
            }

            logger.debug(f"LLM Warmup: Sending warmup request with {len(warmup_messages)} messages...")

            response = requests.post(
                str(self.completion_url),
                headers=headers,
                json=payload,
                timeout=180,  # Allow up to 3 minutes for model loading + prompt processing
            )

            if response.status_code == 200:
                # CRITICAL: Consume and discard the response
                # This ensures even if LLM responded, it won't reach TTS
                result = response.json()
                response_content = result.get("message", {}).get("content", "")
                response_length = len(response_content)

                if response_length > 0:
                    logger.debug(
                        f"LLM Warmup: Model generated {response_length} chars of response "
                        f"(consumed and discarded). First 50 chars: '{response_content[:50]}'"
                    )
                else:
                    logger.debug("LLM Warmup: Model generated empty response (as expected).")

                logger.success(
                    f"LLM Warmup: Model '{self.llm_model}' loaded and system prompt processed. "
                    f"Will remain loaded for {self.keep_alive_timeout}."
                )
            else:
                logger.warning(
                    f"LLM Warmup: Received HTTP {response.status_code} from Ollama. "
                    f"Model may not be pre-loaded, but system will continue."
                )

        except requests.exceptions.Timeout:
            logger.warning(
                f"LLM Warmup: Timeout while loading model '{self.llm_model}'. "
                f"The model is likely large and may take longer on first real request. "
                f"System will continue normally."
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"LLM Warmup: Could not connect to Ollama at {self.completion_url}. "
                f"Please ensure Ollama is running. System will continue but LLM requests may fail."
            )
        except Exception as e:
            logger.warning(
                f"LLM Warmup: Unexpected error during warmup: {e}. "
                f"This is non-critical - system will continue."
            )

    def play_announcement(self, interruptible: bool | None = None) -> None:
        """
        Play the announcement using text-to-speech (TTS) synthesis.

        This method checks if an announcement is set and, if so, places it in the TTS queue for processing.
        If the `interruptible` parameter is set to `True`, it allows the announcement to be interrupted by other
        audio playback. If `interruptible` is `None`, it defaults to the instance's `interruptible` setting.

        Args:
            interruptible (bool | None): Whether the announcement can be interrupted by other audio playback.
                If `None`, it defaults to the instance's `interruptible` setting.
        """

        if interruptible is None:
            interruptible = self.interruptible
        logger.success("Playing announcement...")
        if self.announcement:
            announcement_msg = TTSTextMessage(
                text=self.announcement,
                sequence_num=-1,
                is_eos=False,
            )
            self.tts_queue.put(announcement_msg)
            self.processing_active_event.set()

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Retrieve the current list of conversation messages.

        Returns:
            list[dict[str, str]]: A list of message dictionaries representing the conversation history.
        """
        return self._messages

    @classmethod
    def from_config(cls, config: GladosConfig) -> "Glados":
        """
        Create a Glados instance from a GladosConfig configuration object.

        Supports both monolingual (legacy) and bilingual (EN-Branch) configurations.

        Parameters:
            config (GladosConfig): Configuration object containing Glados initialization parameters

        Returns:
            Glados: A new Glados instance configured with the provided settings
        """

        # Legacy/main models (always required)
        asr_model = get_audio_transcriber(
            engine_type=config.asr_engine,
            language=config.language,
        )

        tts_model: SpeechSynthesizerProtocol
        tts_model = get_speech_synthesizer(config.voice)

        audio_io = get_audio_system(backend_type=config.audio_io)

        # EN-Branch models (optional, only if enabled)
        ru_asr_model = None
        en_asr_model = None
        ru_tts_model = None
        en_tts_model = None
        lid_model = None

        if config.enable_en_branch:
            logger.info("EN-Branch: Loading bilingual mode models from configuration...")

            try:
                # Load Russian ASR
                if config.asr_ru_engine:
                    logger.info(f"EN-Branch: Loading Russian ASR engine: {config.asr_ru_engine}")
                    ru_asr_model = get_audio_transcriber(
                        engine_type=config.asr_ru_engine,
                        language="ru",
                    )

                # Load English ASR
                if config.asr_en_engine:
                    logger.info(f"EN-Branch: Loading English ASR engine: {config.asr_en_engine}")
                    en_asr_model = get_audio_transcriber(
                        engine_type=config.asr_en_engine,
                        language="en",
                    )

                # Load Russian TTS
                if config.voice_ru:
                    logger.info(f"EN-Branch: Loading Russian TTS voice: {config.voice_ru}")
                    ru_tts_model = get_speech_synthesizer(config.voice_ru)

                # Load English TTS
                if config.voice_en:
                    logger.info(f"EN-Branch: Loading English TTS voice: {config.voice_en}")
                    en_tts_model = get_speech_synthesizer(config.voice_en)

                # Load Language ID model
                logger.info("EN-Branch: Loading SpeechBrain Language ID model...")
                from ..audio_io import SpeechBrainLanguageID

                lid_config = config.language_detection or {}
                model_name = lid_config.get("model_name", "speechbrain/lang-id-voxlingua107-ecapa")
                default_language = lid_config.get("default_language", "ru")

                lid_model = SpeechBrainLanguageID(
                    model_name=model_name,
                    device=None,  # Auto-detect CUDA
                    confidence_threshold=lid_config.get("confidence_threshold", 0.7),
                    default_language=default_language,
                )

                logger.success("EN-Branch: All bilingual models loaded successfully!")

            except Exception as e:
                logger.error(f"EN-Branch: Failed to load bilingual models: {e}")
                logger.exception(e)
                logger.warning("EN-Branch: Will attempt to start in monolingual mode.")
                config.enable_en_branch = False

        return cls(
            asr_model=asr_model,
            tts_model=tts_model,
            audio_io=audio_io,
            completion_url=config.completion_url,
            llm_model=config.llm_model,
            api_key=config.api_key,
            keep_alive_timeout=config.keep_alive_timeout,
            enable_thinking=config.enable_thinking,
            thinking_trigger_words=config.thinking_trigger_words,
            thinking_fuzzy_threshold=config.thinking_fuzzy_threshold,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=config.announcement,
            personality_preprompt=tuple(config.to_chat_messages()),
            # EN-Branch parameters
            enable_en_branch=config.enable_en_branch,
            ru_asr_model=ru_asr_model,
            en_asr_model=en_asr_model,
            ru_tts_model=ru_tts_model,
            en_tts_model=en_tts_model,
            lid_model=lid_model,
            language_detection_config=config.language_detection,
            audio_mixer_config=config.audio_mixer,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Glados":
        """
        Create a Glados instance from a configuration file.

        Parameters:
            path (str): Path to the YAML configuration file containing Glados settings.

        Returns:
            Glados: A new Glados instance configured with settings from the specified YAML file.

        Example:
            glados = Glados.from_yaml('config/default.yaml')
        """
        return cls.from_config(GladosConfig.from_yaml(path))

    def run(self) -> None:
        """
        Start the voice assistant's listening event loop, continuously processing audio input.
        This method initializes the audio input system, starts listening for audio samples,
        and enters a loop that waits for audio input until a shutdown event is triggered.
        It handles keyboard interrupts gracefully and ensures that all components are properly shut down.

        This method is the main entry point for running the Glados voice assistant.
        """
        self.audio_io.start_listening()

        logger.success("Audio Modules Operational")
        logger.success("Listening...")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                time.sleep(self.PAUSE_TIME)
            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in main run loop.")
            # Make sure any ongoing audio playback is stopped
            if self.currently_speaking_event.is_set():
                for component in self.component_threads:
                    if component.name == "AudioPlayer":
                        self.audio_io.stop_speaking()
                        self.currently_speaking_event.clear()
                        break

            # End chat session before shutdown
            if hasattr(self, 'chat_logger'):
                self.chat_logger.end_session(reason="manual")

            self.shutdown_event.set()
            # Give threads a moment to notice the shutdown event
            time.sleep(self.PAUSE_TIME)
        finally:
            logger.info("Listen event loop is stopping/exiting.")
            sys.exit(0)


if __name__ == "__main__":
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.run()
