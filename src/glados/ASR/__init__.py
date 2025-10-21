"""ASR processing components."""

from pathlib import Path
from typing import Any, Protocol
import warnings

from numpy.typing import NDArray

from .mel_spectrogram import MelSpectrogramCalculator


class TranscriberProtocol(Protocol):
    def __init__(self, model_path: str, *args: str, **kwargs: dict[str, str]) -> None: ...
    def transcribe(self, audio_source: NDArray[Any]) -> str: ...
    def transcribe_file(self, audio_path: Path) -> str: ...


# Factory function
def get_audio_transcriber(
        engine_type: str = "ctc",
        *,
        language: str = "en",
        **kwargs: Any,
) -> TranscriberProtocol:
    """
    Factory function to get an instance of an audio transcriber based on the specified engine type.

    Parameters:
        engine_type (str): The type of ASR engine to use:
            - "ctc": Connectionist Temporal Classification model (faster, good accuracy)
            - "tdt": Token and Duration Transducer model (best accuracy, slightly slower)
        **kwargs: Additional keyword arguments to pass to the transcriber constructor

    Returns:
        TranscriberProtocol: An instance of the requested audio transcriber

    Raises:
        ValueError: If the specified engine type is not supported
    """
    normalized_language = language.lower()

    requested_engine = engine_type.lower()

    if requested_engine == "gigaam" or normalized_language == "ru":
        try:
            from .gigaam_asr import AudioTranscriber as GigaAMTranscriber
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            warnings.warn(
                "GigaAM ASR backend is unavailable because optional dependencies are missing. "
                "Falling back to the ONNX-based models.",
                stacklevel=2,
            )
        else:
            try:
                return GigaAMTranscriber(language=normalized_language, **kwargs)
            except RuntimeError as exc:  # pragma: no cover - import guard
                warnings.warn(
                    f"Failed to initialize GigaAM ASR backend ({exc}). Falling back to the ONNX-based models.",
                    stacklevel=2,
                )

        requested_engine = "tdt" if requested_engine == "gigaam" else requested_engine

    if requested_engine == "ctc":
        from .ctc_asr import AudioTranscriber as CTCTranscriber

        return CTCTranscriber()
    elif requested_engine == "tdt":
        from .tdt_asr import AudioTranscriber as TDTTranscriber

        return TDTTranscriber()
    else:
        raise ValueError(f"Unsupported ASR engine type: {engine_type}")


__all__ = ["MelSpectrogramCalculator", "TranscriberProtocol", "get_audio_transcriber"]
