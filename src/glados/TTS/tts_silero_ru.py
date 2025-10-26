"""Silero TTS V4 Russian implementation for GLaDOS.

This module provides Russian text-to-speech synthesis using Silero V4 models
with the 'xenia' speaker voice.
"""

import torch
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from pathlib import Path


class SileroRuSynthesizer:
    """Russian TTS synthesizer using Silero V4 models.

    This synthesizer uses the Silero TTS V4 model with the 'xenia' speaker
    for high-quality Russian speech synthesis.
    """

    def __init__(
        self,
        speaker: str = "xenia",
        sample_rate: int = 48000,
        device: str | None = None,
    ):
        """Initialize Silero Russian TTS synthesizer.

        Args:
            speaker: Speaker name (default: 'xenia')
                Available: 'aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'
            sample_rate: Audio sample rate (8000, 24000, or 48000)
            device: Device to run model on ('cpu', 'cuda', or None for auto)

        Note:
            V4 model has automatic stress placement (put_accent) built-in.
        """
        self.speaker = speaker
        self.sample_rate = sample_rate

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Silero Russian TTS on {self.device}")

        try:
            # Load Silero V4 Russian model
            # Returns: (model, example_text)
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru'
            )

            # Move model to device (in-place operation)
            model.to(self.device)

            # Save reference to model
            self.model = model

            logger.success(f"Silero V4 Russian TTS loaded successfully with speaker '{speaker}' on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Silero TTS model: {e}")
            raise

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Generate speech audio from Russian text.

        Args:
            text: Russian text to synthesize

        Returns:
            Audio samples as float32 numpy array
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return np.array([], dtype=np.float32)

        try:
            # Generate audio using Silero V4
            # Note: V4 has automatic stress (put_accent) built-in
            with torch.no_grad():
                audio_tensor = self.model.apply_tts(
                    text=text,
                    speaker=self.speaker,
                    sample_rate=self.sample_rate
                )

            # Convert to numpy array
            if isinstance(audio_tensor, torch.Tensor):
                audio = audio_tensor.cpu().numpy()
            else:
                audio = np.array(audio_tensor, dtype=np.float32)

            # Ensure it's 1D
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Ensure float32
            audio = audio.astype(np.float32)

            logger.debug(f"Generated {len(audio)/self.sample_rate:.2f}s of audio for text: '{text[:50]}...'")

            return audio

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            return np.array([], dtype=np.float32)

    def save_audio(self, audio: NDArray[np.float32], path: Path | str) -> None:
        """Save audio to WAV file.

        Args:
            audio: Audio samples
            path: Output file path
        """
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(path, audio, self.sample_rate)
        logger.info(f"Audio saved to {path}")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
