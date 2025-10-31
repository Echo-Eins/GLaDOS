"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
def get_speech_synthesizer(
    voice: str = "glados",
) -> SpeechSynthesizerProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer (English)
            - "glados_ru": GLaDOS voice synthesizer (Russian with Silero TTS + RVC + Audio Processing)
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados

        return tts_glados.SpeechSynthesizer()

    if voice.lower() == "glados_ru":
        from ..TTS import tts_glados_ru

        return tts_glados_ru.GLaDOSRuSynthesizer()

    if voice.lower() == "silero_ru":
        from ..TTS import tts_silero_ru

        return tts_silero_ru.SileroRuSynthesizer()

    if voice.lower() == "silero_en":
        from ..TTS import tts_silero_en

        return tts_silero_en.SileroEnSynthesizer()

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
