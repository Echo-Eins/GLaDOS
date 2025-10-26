"""Audio processing modules for GLaDOS voice synthesis."""

from .audio_processor import AudioEQProcessor, Compressor, Reverb, AudioProcessingPipeline
from .preset_manager import PresetManager
from .rvc_processor import RVCProcessor, create_rvc_processor
from .audio_tui import run_audio_processor_tui, AudioProcessorTUI

__all__ = [
    "AudioEQProcessor",
    "Compressor",
    "Reverb",
    "AudioProcessingPipeline",
    "PresetManager",
    "RVCProcessor",
    "create_rvc_processor",
    "run_audio_processor_tui",
    "AudioProcessorTUI",
]
