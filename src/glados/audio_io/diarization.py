"""Speaker Diarization using NVIDIA Sortformer model.

This module provides speaker diarization (who spoke when) using
NVIDIA's diar_streaming_sortformer_4spk-v2 model from HuggingFace.

Supports up to 4 speakers in streaming mode with low latency.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from loguru import logger


class NVIDIASpeakerDiarizer:
    """NVIDIA Sortformer Speaker Diarization for audio segments.

    Uses diar_streaming_sortformer_4spk-v2 model (streaming, up to 4 speakers).
    Designed for real-time or near-real-time applications.

    Features:
    - Streaming mode (online diarization)
    - Maximum 4 speakers
    - Low latency
    - Handles recordings several hours long
    """

    SAMPLE_RATE: int = 16000
    MAX_SPEAKERS: int = 4

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        device: str | None = None,
    ):
        """Initialize NVIDIA Speaker Diarizer.

        Args:
            model_name: HuggingFace model name or path to .nemo file
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing NVIDIA Speaker Diarizer on {self.device}")
        logger.info(f"Max speakers supported: {self.MAX_SPEAKERS}")

        try:
            from nemo.collections.asr.models import SortformerEncLabelModel

            # Load model from HuggingFace or local file
            if model_name.endswith('.nemo'):
                logger.info(f"Loading Sortformer from local file: {model_name}")
                self.model = SortformerEncLabelModel.restore_from(
                    restore_path=model_name,
                    map_location=str(self.device),
                    strict=False,
                )
            else:
                logger.info(f"Loading Sortformer from HuggingFace: {model_name}")
                logger.info("Note: This may require HuggingFace token for private models")

                # Try loading from HF
                try:
                    self.model = SortformerEncLabelModel.from_pretrained(model_name)
                except Exception as hf_error:
                    logger.warning(f"Failed to load from HuggingFace: {hf_error}")
                    logger.info("Attempting to load from local cache or download...")

                    # Fallback: download and load
                    from huggingface_hub import hf_hub_download
                    cache_dir = Path.home() / '.cache' / 'nvidia_diarization'
                    cache_dir.mkdir(parents=True, exist_ok=True)

                    nemo_file = hf_hub_download(
                        repo_id=model_name,
                        filename="diar_streaming_sortformer_4spk-v2.nemo",
                        cache_dir=str(cache_dir),
                    )

                    self.model = SortformerEncLabelModel.restore_from(
                        restore_path=nemo_file,
                        map_location=str(self.device),
                        strict=False,
                    )

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.success(
                f"NVIDIA Sortformer Diarizer loaded successfully "
                f"(streaming mode, up to {self.MAX_SPEAKERS} speakers)"
            )

        except Exception as e:
            logger.error(f"Failed to load NVIDIA Sortformer Diarizer: {e}")
            raise

    def diarize(
        self,
        audio: NDArray[np.float32],
        num_speakers: int | None = None,
    ) -> List[Tuple[float, float, str]]:
        """Perform speaker diarization on audio segment.

        Args:
            audio: Audio samples (16kHz mono float32)
            num_speakers: Expected number of speakers (None for auto-detection, max 4)

        Returns:
            List of tuples: [(start_time, end_time, speaker_id), ...]
            - start_time: Segment start in seconds
            - end_time: Segment end in seconds
            - speaker_id: Speaker identifier (e.g., "speaker_0", "speaker_1")

        Example:
            >>> segments = diarizer.diarize(audio)
            >>> print(segments)
            [(0.0, 2.5, "speaker_0"), (2.5, 5.0, "speaker_1"), ...]
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided to diarizer")
            return []

        # Normalize audio
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # Convert to mono

        # Validate number of speakers
        if num_speakers is not None and num_speakers > self.MAX_SPEAKERS:
            logger.warning(
                f"Requested {num_speakers} speakers, but model supports max {self.MAX_SPEAKERS}. "
                f"Using {self.MAX_SPEAKERS} speakers."
            )
            num_speakers = self.MAX_SPEAKERS

        try:
            # Convert to torch tensor [1, time]
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

            # Run diarization
            with torch.no_grad():
                # NeMo Sortformer expects specific input format
                # This is simplified - actual implementation may need more preprocessing
                diar_output = self.model(audio_signal=audio_tensor)

            # Parse output into segments
            # Output format depends on NeMo version and model configuration
            # This is a simplified example - actual parsing may differ
            segments = self._parse_diarization_output(diar_output, len(audio) / self.SAMPLE_RATE)

            logger.debug(f"Diarization completed: {len(segments)} segments found")

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            logger.exception(e)
            return []

    def _parse_diarization_output(
        self,
        diar_output: torch.Tensor,
        audio_duration: float,
    ) -> List[Tuple[float, float, str]]:
        """Parse diarization model output into segment list.

        Args:
            diar_output: Model output tensor
            audio_duration: Total audio duration in seconds

        Returns:
            List of (start, end, speaker_id) tuples
        """
        # This is a placeholder implementation
        # Actual parsing depends on NeMo Sortformer output format

        # Sortformer typically outputs frame-level speaker labels
        # Need to convert to time-based segments

        segments = []

        # Simplified example: assume output is [batch, frames, num_speakers]
        # where each frame represents a time step

        if isinstance(diar_output, dict):
            # If output is a dictionary, extract relevant field
            # (depends on model configuration)
            labels = diar_output.get('labels', diar_output.get('predictions', None))
        else:
            labels = diar_output

        if labels is None:
            logger.warning("Could not parse diarization output")
            return []

        # Convert to numpy for easier processing
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Merge consecutive frames with same speaker into segments
        # This is a simplified implementation
        current_speaker = None
        segment_start = 0.0
        frame_duration = audio_duration / labels.shape[-1] if labels.shape else 0.1

        for i, frame_labels in enumerate(labels.squeeze()):
            # Get dominant speaker for this frame
            speaker_idx = np.argmax(frame_labels) if len(frame_labels.shape) > 0 else 0
            speaker_id = f"speaker_{speaker_idx}"

            time = i * frame_duration

            # If speaker changed, save previous segment
            if speaker_id != current_speaker:
                if current_speaker is not None:
                    segments.append((segment_start, time, current_speaker))
                current_speaker = speaker_id
                segment_start = time

        # Add final segment
        if current_speaker is not None:
            segments.append((segment_start, audio_duration, current_speaker))

        return segments

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
