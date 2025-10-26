"""Audio backend abstractions for Glados."""

import os
import queue
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .vad import VAD


class AudioProtocol(Protocol):
    def __init__(self, vad_threshold: float | None = None) -> None: ...
    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...
    def start_speaking(
        self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = ""
    ) -> None: ...
    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]: ...
    def check_if_speaking(self) -> bool: ...
    def stop_speaking(self) -> None: ...
    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]: ...


# Factory function
def get_audio_system(backend_type: str = "sounddevice", vad_threshold: float | None = None) -> AudioProtocol:
    """Return an audio backend implementation."""

    fallback_backend = os.getenv("GLADOS_AUDIO_FALLBACK")

    if backend_type == "sounddevice":
        if fallback_backend == "null":
            from .null_io import NullAudioIO

            return NullAudioIO(vad_threshold=vad_threshold)

        from .sounddevice_io import SoundDeviceAudioIO

        return SoundDeviceAudioIO(
            vad_threshold=vad_threshold,
        )
    elif backend_type == "websocket":
        raise ValueError("WebSocket audio backend is not yet implemented.")
    elif backend_type == "null":
        from .null_io import NullAudioIO

        return NullAudioIO(vad_threshold=vad_threshold)
    else:
        raise ValueError(f"Unsupported audio backend type: {backend_type}")


__all__ = [
    "VAD",
    "AudioProtocol",
    "get_audio_system",
]

