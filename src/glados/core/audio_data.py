"""Core audio data structures for GLaDOS voice assistant.

This module defines message classes used for audio processing and communication
between different components of the voice assistant pipeline.
"""

from dataclasses import dataclass

from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    """Audio message container for TTS output.

    Args:
        audio: Generated audio samples as float32 array
        text: Associated text that was synthesized
        is_eos: Flag indicating end of speech stream
        sequence_num: Sequence number for ordered playback (0 = play immediately)
    """

    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
    sequence_num: int = 0


@dataclass
class AudioInputMessage:
    """Audio input message container for ASR processing.

    Args:
        audio_sample: Raw audio input samples as float32 array
        vad_confidence: Voice activity detection confidence flag
    """

    audio_sample: NDArray[np.float32]
    vad_confidence: bool = False

@dataclass
class RecognitionResult:
    """Speech recognition outcome including optional emotion probabilities."""

    text: str
    emotions: Mapping[str, float] | None = None


@dataclass
class TTSTextMessage:
    """Text message for TTS processing with sequence number.

    Args:
        text: Text to synthesize
        sequence_num: Sequence number for preserving playback order
        is_eos: Flag indicating end of stream
    """

    text: str
    sequence_num: int = 0
    is_eos: bool = False
