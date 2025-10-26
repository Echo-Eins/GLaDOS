"""RVC (Retrieval-based Voice Conversion) processor for GLaDOS voice.

This module handles voice conversion using pre-trained RVC models
to transform input speech into GLaDOS-like voice.

Uses inferrvc (CircuitCM/RVC-inference) for high-performance RVC inference
with Python 3.12 support.
"""

import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from loguru import logger

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available. RVC processing will be limited.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. RVC index search will be disabled.")

# IMPORTANT: Lazy import of inferrvc to avoid argparse conflict!
# inferrvc has argparse.parse_args() in Config.__init__ which runs on import
# and conflicts with glados CLI arguments. We check availability without importing.
INFERRVC_AVAILABLE = False
try:
    import importlib.util
    spec = importlib.util.find_spec("inferrvc")
    if spec is not None:
        INFERRVC_AVAILABLE = True
        logger.info("inferrvc library detected (lazy import)")
except ImportError:
    logger.warning("inferrvc not available. RVC will use fallback mode.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


class RVCProcessor:
    """RVC voice conversion processor for GLaDOS voice.

    Converts input audio to GLaDOS voice using pre-trained RVC model.
    Uses inferrvc library for high-performance RVC inference.
    """

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        device: str | None = None,
        f0_method: str = "rmvpe",
        f0_up_key: int = 0,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        protect: float = 0.33,
    ):
        """Initialize RVC processor.

        Args:
            model_path: Path to RVC model (.pth file)
            index_path: Path to feature index (.index file)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            f0_method: F0 extraction method ('rmvpe', 'harvest', 'pm', 'dio', 'crepe')
            f0_up_key: Pitch shift in semitones (-12 to +12)
            index_rate: Feature index influence (0.0 to 1.0)
            filter_radius: Median filter radius for F0 smoothing (0-7)
            protect: Consonant protection (0.0 to 0.5)
        """
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.f0_method = f0_method
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.protect = protect

        # Determine device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing RVC processor on {self.device}")

        # Check model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        # Initialize RVC inference
        self.rvc = None
        if INFERRVC_AVAILABLE:
            self._init_inferrvc()
        else:
            logger.warning("RVC will run in fallback mode (no voice conversion)")

        logger.success("RVC processor initialized")

    def _init_inferrvc(self) -> None:
        """Initialize inferrvc inference engine."""
        try:
            logger.info("Initializing inferrvc inference...")

            # Lazy import inferrvc HERE to avoid argparse conflict
            # inferrvc has argparse in Config.__init__ that runs on import
            # and tries to parse sys.argv, which contains GLaDOS arguments!
            # Solution: Temporarily clear sys.argv during import
            import sys
            original_argv = sys.argv.copy()
            sys.argv = [sys.argv[0]]  # Keep only script name

            try:
                from inferrvc import RVC as InferRVC
                self.InferRVC = InferRVC  # Store class reference
            finally:
                # Restore original argv
                sys.argv = original_argv

            # Determine index path
            if self.index_path and self.index_path.exists():
                index_str = str(self.index_path)
                logger.info(f"Using feature index: {self.index_path}")
            else:
                # inferrvc can auto-detect index with same name as model
                index_name = self.model_path.stem  # filename without .pth
                index_str = index_name
                logger.info(f"Attempting auto-detect index: {index_name}")

            # Initialize inferrvc RVC model
            # inferrvc automatically uses FP16 on CUDA for better performance (3x faster)
            # is_half is controlled via Config, not RVC.__init__ parameter
            # Just pass the device and inferrvc will configure FP16 automatically
            self.rvc = InferRVC(
                model=str(self.model_path),
                index=index_str
            )

            # Check if FP16 is enabled (inferrvc auto-detects based on device)
            self.is_half = getattr(self.rvc.config, 'is_half', False)

            logger.success(f"inferrvc initialized successfully: {self.rvc.name}")
            logger.info(f"Model sample rate: {self.rvc.tgt_sr}Hz, version: {self.rvc.version}")
            logger.info(f"Using {'FP16' if self.is_half else 'FP32'} precision on {self.rvc.config.device}")

        except Exception as e:
            logger.error(f"Failed to initialize inferrvc: {e}")
            logger.exception(e)
            self.rvc = None

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio through RVC voice conversion.

        Args:
            audio: Input audio samples (float32, -1.0 to 1.0)
            input_sample_rate: Sample rate of input audio

        Returns:
            Converted audio samples (float32)
        """
        if len(audio) == 0:
            return audio

        # If RVC not available, return original audio
        if self.rvc is None:
            logger.warning("RVC not available - returning original audio")
            return audio

        try:
            audio_duration = len(audio) / input_sample_rate
            logger.debug(f"RVC processing: {audio_duration:.2f}s audio")

            # Convert to torch tensor (fp32 initially for resampling compatibility)
            audio_tensor = torch.from_numpy(audio).float()

            # Resample to 16kHz (required by RVC)
            # Keep in fp32 for resampling to avoid dtype issues with torchaudio
            if input_sample_rate != 16000:
                import torchaudio
                # Create resampler in fp32 to avoid kernel dtype mismatch
                resampler = torchaudio.transforms.Resample(
                    orig_freq=input_sample_rate,
                    new_freq=16000
                )
                audio_tensor = resampler(audio_tensor)
                logger.debug(f"Resampled from {input_sample_rate}Hz to 16kHz")

            # Normalize if needed
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
                logger.debug(f"Normalized audio (max={max_val:.3f})")

            # inferrvc uses t_pad for reflection padding
            # t_pad = tgt_sr (typically 48000 at model's target sample rate)
            # At 16kHz input, t_pad is scaled: 48000 * (16000/48000) = 16000 samples = 1 second
            # For reflection padding, we need audio length > 2*t_pad
            t_pad_16k = int(self.rvc.tgt_sr * (16000 / self.rvc.tgt_sr))  # t_pad in 16kHz
            min_samples = 2 * t_pad_16k + 1600  # Add extra 0.1s buffer

            original_length = len(audio_tensor)
            added_padding = 0

            if original_length < min_samples:
                # Add silence padding at the end to reach minimum length
                added_padding = min_samples - original_length
                silence = torch.zeros(added_padding, dtype=audio_tensor.dtype)
                audio_tensor = torch.cat([audio_tensor, silence])
                logger.debug(
                    f"Added {added_padding/16000:.2f}s silence padding "
                    f"(min required: {min_samples/16000:.2f}s, audio: {original_length/16000:.2f}s)"
                )

            # Process with inferrvc
            logger.debug(
                f"Running RVC inference: {self.f0_method}, "
                f"pitch={self.f0_up_key}, index_rate={self.index_rate}"
            )

            # Get InferRVC class reference (stored during init)
            InferRVC = getattr(self, 'InferRVC', None)
            if InferRVC is None:
                # Fallback: import if not already imported (shouldn't happen)
                import sys
                original_argv = sys.argv.copy()
                sys.argv = [sys.argv[0]]
                try:
                    from inferrvc import RVC as InferRVC
                finally:
                    sys.argv = original_argv

            # inferrvc has FP16 compatibility bugs with torchaudio operations:
            # Many torchaudio ops use native C++ kernels that only support FP32/FP64
            # This includes: Resample, Loudness, biquad filters, lfilter, pitch_shift, etc.
            # Universal solution: Wrap ALL forward() methods in torchaudio.transforms
            # to auto-convert FP16 <-> FP32 as needed
            # Only needed when is_half=True (FP16 mode)
            patched_classes = []
            original_methods = {}
            original_resample = None

            if self.is_half:
                import inferrvc.modules
                import torchaudio.transforms
                import inspect

                # Patch 1: ResampleCache for internal resampling (not a transform)
                original_resample = inferrvc.modules.ResampleCache.resample

                @staticmethod
                def patched_resample(fromto, audio, deviceto):
                    """Patched resample that handles FP16/FP32 conversion."""
                    was_half = audio.dtype == torch.float16
                    if was_half:
                        audio = audio.float()
                    result = original_resample(fromto, audio, deviceto)
                    if was_half and deviceto.startswith('cuda'):
                        result = result.half()
                    return result

                inferrvc.modules.ResampleCache.resample = patched_resample

                # Patch 2: Universal wrapper for ALL torchaudio transforms
                # Get all transform classes from torchaudio.transforms
                for name in dir(torchaudio.transforms):
                    obj = getattr(torchaudio.transforms, name)
                    # Check if it's a class (not function/module) and has forward method
                    if inspect.isclass(obj) and hasattr(obj, 'forward'):
                        # Store original forward method
                        original_forward = obj.forward
                        original_methods[name] = original_forward

                        def make_patched_forward(original_fn):
                            """Create a patched forward that handles FP16 conversion."""
                            def patched_forward(self, waveform, *args, **kwargs):
                                # Convert to FP32 if needed
                                was_half = isinstance(waveform, torch.Tensor) and waveform.dtype == torch.float16
                                device = waveform.device if isinstance(waveform, torch.Tensor) else None

                                if was_half:
                                    waveform = waveform.float()

                                # Call original forward with FP32
                                result = original_fn(self, waveform, *args, **kwargs)

                                # Convert back to FP16 if needed
                                if was_half and device and device.type == 'cuda':
                                    if isinstance(result, torch.Tensor):
                                        result = result.half()
                                    elif isinstance(result, tuple):
                                        result = tuple(r.half() if isinstance(r, torch.Tensor) else r for r in result)

                                return result
                            return patched_forward

                        # Apply patch
                        obj.forward = make_patched_forward(original_forward)
                        patched_classes.append((obj, name))

                logger.debug(f"Applied FP16->FP32 patches to {len(patched_classes)} torchaudio transforms")

            output_tensor = self.rvc(
                audio_tensor,
                f0_up_key=self.f0_up_key,
                f0_method=self.f0_method,
                index_rate=self.index_rate,
                filter_radius=self.filter_radius,
                protect=self.protect,
                output_device='cpu',  # Always return on CPU
                output_volume=InferRVC.MATCH_ORIGINAL,  # Match input loudness
            )

            # Restore all original functions if we patched them
            if self.is_half and (original_resample or patched_classes):
                # Restore ResampleCache
                if original_resample:
                    inferrvc.modules.ResampleCache.resample = original_resample

                # Restore all torchaudio transforms
                if patched_classes:
                    for obj, name in patched_classes:
                        obj.forward = original_methods[name]
                    logger.debug(f"Restored {len(patched_classes)} torchaudio transforms to original")

            # Convert back to numpy
            converted_audio = output_tensor.cpu().numpy().astype(np.float32)

            # Get output sample rate from inferrvc (default 44100)
            output_sr = getattr(self.rvc, 'outputfreq', 44100)
            logger.debug(f"RVC output: {len(converted_audio)/output_sr:.2f}s at {output_sr}Hz")

            # Remove added padding (scale proportionally to output sample rate)
            if added_padding > 0:
                # Calculate how much padding was added in the output sample rate
                # added_padding is in 16kHz, need to scale to output_sr
                padding_samples_output = int(added_padding * (output_sr / 16000))
                # Trim the silence we added at the end
                original_length_output = len(converted_audio) - padding_samples_output
                converted_audio = converted_audio[:original_length_output]
                logger.debug(
                    f"Removed {padding_samples_output/output_sr:.2f}s padding from output "
                    f"({len(converted_audio)/output_sr:.2f}s remaining)"
                )

            # Resample to match input sample rate if needed
            if output_sr != input_sample_rate and LIBROSA_AVAILABLE:
                logger.debug(f"Resampling from {output_sr}Hz to {input_sample_rate}Hz")
                converted_audio = librosa.resample(
                    converted_audio,
                    orig_sr=output_sr,
                    target_sr=input_sample_rate
                )

            logger.debug(f"RVC conversion complete: {len(converted_audio)/input_sample_rate:.2f}s")
            return converted_audio

        except Exception as e:
            logger.error(f"RVC processing failed: {e}")
            logger.exception(e)
            return audio  # Return original audio on error

    def update_params(
        self,
        f0_up_key: int | None = None,
        f0_method: str | None = None,
        index_rate: float | None = None,
        filter_radius: int | None = None,
        protect: float | None = None,
    ) -> None:
        """Update RVC parameters.

        Args:
            f0_up_key: Pitch shift in semitones
            f0_method: F0 extraction method
            index_rate: Feature index influence
            filter_radius: Median filter radius
            protect: Consonant protection
        """
        if f0_up_key is not None:
            self.f0_up_key = f0_up_key
        if f0_method is not None:
            self.f0_method = f0_method
        if index_rate is not None:
            self.index_rate = index_rate
        if filter_radius is not None:
            self.filter_radius = filter_radius
        if protect is not None:
            self.protect = protect

        logger.debug("RVC parameters updated")


class SimpleRVCProcessor:
    """Simplified RVC processor with basic pitch shifting.

    This is a fallback implementation when full RVC is not available.
    """

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        device: str | None = None,
        pitch_shift: int = 0,
    ):
        """Initialize simple RVC processor.

        Args:
            model_path: Path to RVC model (not used in simple mode)
            index_path: Path to feature index (not used)
            device: Device to run on (not used)
            pitch_shift: Pitch shift in semitones
        """
        self.model_path = Path(model_path)
        self.pitch_shift = pitch_shift

        logger.warning(
            "SimpleRVCProcessor initialized - this only does basic pitch shifting, "
            "not real voice conversion. Install inferrvc for full RVC support."
        )

    def process(
        self,
        audio: NDArray[np.float32],
        input_sample_rate: int = 48000,
    ) -> NDArray[np.float32]:
        """Process audio with basic pitch shifting.

        Args:
            audio: Input audio
            input_sample_rate: Input sample rate

        Returns:
            Pitch-shifted audio
        """
        if len(audio) == 0 or self.pitch_shift == 0:
            return audio

        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, cannot apply pitch shift")
            return audio

        try:
            import librosa
            logger.debug(f"Applying pitch shift: {self.pitch_shift} semitones")
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=input_sample_rate,
                n_steps=self.pitch_shift,
            )
            return shifted.astype(np.float32)
        except Exception as e:
            logger.error(f"Pitch shift failed: {e}")
            return audio


def create_rvc_processor(
    model_path: Path | str,
    index_path: Path | str | None = None,
    simple_mode: bool = False,
    **kwargs
) -> RVCProcessor | SimpleRVCProcessor:
    """Factory function to create appropriate RVC processor.

    Args:
        model_path: Path to RVC model
        index_path: Path to feature index
        simple_mode: Use simplified processor
        **kwargs: Additional arguments for processor

    Returns:
        RVC processor instance
    """
    if simple_mode or not INFERRVC_AVAILABLE:
        if not simple_mode:
            logger.warning("inferrvc not available, using simple mode")
        return SimpleRVCProcessor(
            model_path,
            index_path,
            pitch_shift=kwargs.get('f0_up_key', 0),
            **{k: v for k, v in kwargs.items() if k in ['device']}
        )
    else:
        return RVCProcessor(model_path, index_path, **kwargs)
