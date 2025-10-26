"""Advanced audio processing module for GLaDOS voice synthesis.

This module provides parametric EQ, compression, and reverb effects
to achieve the characteristic GLaDOS timbre.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from loguru import logger


@dataclass
class EQBand:
    """Parametric EQ band configuration."""

    frequency: float
    """Center/cutoff frequency in Hz"""

    gain_db: float
    """Gain in dB (positive = boost, negative = cut)"""

    q_factor: float
    """Quality factor (bandwidth)"""

    filter_type: Literal['lowpass', 'highpass', 'peak', 'lowshelf', 'highshelf']
    """Filter type"""


class AudioEQProcessor:
    """Parametric equalizer using IIR biquad filters.

    Supports multiple bands with different filter types for precise
    frequency shaping to achieve GLaDOS-like timbre.
    """

    def __init__(self, sample_rate: int = 48000):
        """Initialize the EQ processor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.bands: list[EQBand] = []
        self._filters: list[tuple[NDArray, NDArray]] = []

    def add_band(
        self,
        freq: float,
        gain_db: float,
        q: float,
        filter_type: Literal['lowpass', 'highpass', 'peak', 'lowshelf', 'highshelf']
    ) -> None:
        """Add an EQ band to the processor.

        Args:
            freq: Center/cutoff frequency in Hz
            gain_db: Gain in dB
            q: Quality factor (Q)
            filter_type: Type of filter to apply
        """
        band = EQBand(frequency=freq, gain_db=gain_db, q_factor=q, filter_type=filter_type)
        self.bands.append(band)
        self._update_filters()

    def _update_filters(self) -> None:
        """Recalculate biquad filter coefficients for all bands."""
        self._filters = []

        for band in self.bands:
            # Normalize frequency to Nyquist
            freq_normalized = band.frequency / (self.sample_rate / 2)

            # Ensure frequency is in valid range
            freq_normalized = np.clip(freq_normalized, 0.001, 0.999)

            if band.filter_type == 'lowpass':
                b, a = signal.butter(2, freq_normalized, btype='low')
            elif band.filter_type == 'highpass':
                b, a = signal.butter(2, freq_normalized, btype='high')
            elif band.filter_type == 'peak':
                # Peaking EQ (parametric)
                b, a = self._peaking_eq(freq_normalized, band.gain_db, band.q_factor)
            elif band.filter_type == 'lowshelf':
                b, a = self._shelving_eq(freq_normalized, band.gain_db, 'low')
            elif band.filter_type == 'highshelf':
                b, a = self._shelving_eq(freq_normalized, band.gain_db, 'high')
            else:
                raise ValueError(f"Unknown filter type: {band.filter_type}")

            self._filters.append((b, a))

    def _peaking_eq(self, freq_norm: float, gain_db: float, q: float) -> tuple[NDArray, NDArray]:
        """Design a peaking EQ filter.

        Args:
            freq_norm: Normalized frequency (0 to 1)
            gain_db: Gain in dB
            q: Quality factor

        Returns:
            Tuple of (b, a) filter coefficients
        """
        A = 10 ** (gain_db / 40)
        omega = np.pi * freq_norm
        alpha = np.sin(omega) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(omega)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha / A

        return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0

    def _shelving_eq(
        self,
        freq_norm: float,
        gain_db: float,
        shelf_type: Literal['low', 'high']
    ) -> tuple[NDArray, NDArray]:
        """Design a shelving EQ filter.

        Args:
            freq_norm: Normalized frequency (0 to 1)
            gain_db: Gain in dB
            shelf_type: 'low' or 'high'

        Returns:
            Tuple of (b, a) filter coefficients
        """
        A = 10 ** (gain_db / 40)
        omega = np.pi * freq_norm
        alpha = np.sin(omega) / 2 * np.sqrt((A + 1/A) * (1/0.7 - 1) + 2)

        if shelf_type == 'low':
            b0 = A * ((A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega))
            b2 = A * ((A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(omega))
            a2 = (A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha
        else:  # high
            b0 = A * ((A + 1) + (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega))
            b2 = A * ((A + 1) + (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(omega) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(omega))
            a2 = (A + 1) - (A - 1) * np.cos(omega) - 2 * np.sqrt(A) * alpha

        return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0

    def apply(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply all EQ bands to the audio signal.

        Args:
            audio: Input audio signal

        Returns:
            Processed audio signal
        """
        if len(self._filters) == 0:
            return audio

        output = audio.copy()

        for b, a in self._filters:
            output = signal.filtfilt(b, a, output).astype(np.float32)

        return output

    def clear_bands(self) -> None:
        """Remove all EQ bands."""
        self.bands.clear()
        self._filters.clear()

    def to_dict(self) -> dict:
        """Export EQ configuration to dictionary.

        Returns:
            Dictionary representation of all EQ bands
        """
        return [
            {
                "type": band.filter_type,
                "frequency": band.frequency,
                "gain_db": band.gain_db,
                "q_factor": band.q_factor,
            }
            for band in self.bands
        ]

    def from_dict(self, config: list[dict]) -> None:
        """Load EQ configuration from dictionary.

        Args:
            config: List of band configurations
        """
        self.clear_bands()
        for band_config in config:
            self.add_band(
                freq=band_config["frequency"],
                gain_db=band_config["gain_db"],
                q=band_config["q_factor"],
                filter_type=band_config["type"]
            )


class Compressor:
    """RMS-based dynamic range compressor.

    Reduces the dynamic range of audio by attenuating signals
    above a threshold, making loud parts quieter.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        makeup_gain_db: float = 0.0,
    ):
        """Initialize the compressor.

        Args:
            sample_rate: Audio sample rate in Hz
            threshold_db: Threshold in dB (signals above this are compressed)
            ratio: Compression ratio (e.g., 4.0 means 4:1)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            makeup_gain_db: Makeup gain in dB to compensate for level reduction
        """
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_gain_db = makeup_gain_db

        # Convert times to coefficients
        self.attack_coef = np.exp(-1000.0 / (attack_ms * sample_rate))
        self.release_coef = np.exp(-1000.0 / (release_ms * sample_rate))

    def apply(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply compression to the audio signal.

        Args:
            audio: Input audio signal

        Returns:
            Compressed audio signal
        """
        if len(audio) == 0:
            return audio

        # Convert to dB
        epsilon = 1e-10
        audio_abs = np.abs(audio) + epsilon
        audio_db = 20 * np.log10(audio_abs)

        # Calculate gain reduction
        gain_reduction_db = np.zeros_like(audio_db)
        envelope_db = audio_db[0]

        for i in range(len(audio_db)):
            # Envelope follower
            if audio_db[i] > envelope_db:
                envelope_db = self.attack_coef * envelope_db + (1 - self.attack_coef) * audio_db[i]
            else:
                envelope_db = self.release_coef * envelope_db + (1 - self.release_coef) * audio_db[i]

            # Calculate gain reduction
            if envelope_db > self.threshold_db:
                gain_reduction_db[i] = (self.threshold_db - envelope_db) * (1 - 1/self.ratio)

        # Apply gain reduction and makeup gain
        total_gain_db = gain_reduction_db + self.makeup_gain_db
        gain_linear = 10 ** (total_gain_db / 20)

        output = audio * gain_linear

        # Prevent clipping
        output = np.clip(output, -1.0, 1.0)

        return output.astype(np.float32)

    def to_dict(self) -> dict:
        """Export compressor configuration to dictionary."""
        return {
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "makeup_gain_db": self.makeup_gain_db,
        }

    def from_dict(self, config: dict) -> None:
        """Load compressor configuration from dictionary."""
        self.threshold_db = config["threshold_db"]
        self.ratio = config["ratio"]
        self.attack_ms = config["attack_ms"]
        self.release_ms = config["release_ms"]
        self.makeup_gain_db = config["makeup_gain_db"]

        # Recalculate coefficients
        self.attack_coef = np.exp(-1000.0 / (self.attack_ms * self.sample_rate))
        self.release_coef = np.exp(-1000.0 / (self.release_ms * self.sample_rate))


class Reverb:
    """Schroeder reverb implementation.

    Creates artificial reverberation using a combination of
    comb and allpass filters.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        decay_s: float = 2.0,
        pre_delay_ms: float = 20.0,
        mix: float = 0.3,
        damping: float = 0.5,
        room_size: float = 0.5,
    ):
        """Initialize the reverb processor.

        Args:
            sample_rate: Audio sample rate in Hz
            decay_s: Reverb decay time in seconds
            pre_delay_ms: Pre-delay before reverb in milliseconds
            mix: Wet/dry mix (0.0 = dry, 1.0 = wet)
            damping: High-frequency damping factor (0.0 to 1.0)
            room_size: Virtual room size factor (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.decay_s = decay_s
        self.pre_delay_ms = pre_delay_ms
        self.mix = mix
        self.damping = damping
        self.room_size = room_size

        # Comb filter delays (in samples) - prime numbers scaled by room size
        base_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        self.comb_delays = [int(d * room_size) for d in base_delays]

        # Allpass filter delays
        self.allpass_delays = [225, 556, 441, 341]

        # Initialize filter buffers
        self.comb_buffers = [np.zeros(d) for d in self.comb_delays]
        self.allpass_buffers = [np.zeros(d) for d in self.allpass_delays]

        # Calculate feedback coefficients for decay time
        self.feedback = 0.84  # Base feedback for roughly 2s decay at 48kHz

    def apply(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply reverb to the audio signal.

        Args:
            audio: Input audio signal

        Returns:
            Reverberated audio signal
        """
        if len(audio) == 0:
            return audio

        # Pre-delay
        pre_delay_samples = int(self.pre_delay_ms * self.sample_rate / 1000)
        if pre_delay_samples > 0:
            delayed = np.concatenate([np.zeros(pre_delay_samples), audio])
        else:
            delayed = audio.copy()

        # Apply parallel comb filters
        comb_out = np.zeros(len(delayed))
        for i, (delay, buffer) in enumerate(zip(self.comb_delays, self.comb_buffers)):
            comb_out += self._comb_filter(delayed, delay, buffer, self.feedback)

        # Apply series allpass filters
        allpass_out = comb_out
        for delay, buffer in zip(self.allpass_delays, self.allpass_buffers):
            allpass_out = self._allpass_filter(allpass_out, delay, buffer)

        # Apply damping (simple lowpass)
        if self.damping > 0:
            allpass_out = self._apply_damping(allpass_out, self.damping)

        # Mix wet and dry
        if len(allpass_out) > len(audio):
            allpass_out = allpass_out[:len(audio)]
        elif len(allpass_out) < len(audio):
            allpass_out = np.pad(allpass_out, (0, len(audio) - len(allpass_out)))

        output = (1 - self.mix) * audio + self.mix * allpass_out

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output.astype(np.float32)

    def _comb_filter(
        self,
        audio: NDArray[np.float32],
        delay: int,
        buffer: NDArray[np.float32],
        feedback: float
    ) -> NDArray[np.float32]:
        """Apply a comb filter."""
        output = np.zeros(len(audio))

        for i in range(len(audio)):
            buffer_idx = i % delay
            output[i] = audio[i] + feedback * buffer[buffer_idx]
            buffer[buffer_idx] = output[i]

        return output

    def _allpass_filter(
        self,
        audio: NDArray[np.float32],
        delay: int,
        buffer: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Apply an allpass filter."""
        output = np.zeros(len(audio))
        feedback = 0.5

        for i in range(len(audio)):
            buffer_idx = i % delay
            delayed = buffer[buffer_idx]
            output[i] = -audio[i] + delayed
            buffer[buffer_idx] = audio[i] + feedback * delayed

        return output

    def _apply_damping(self, audio: NDArray[np.float32], damping: float) -> NDArray[np.float32]:
        """Apply high-frequency damping."""
        # Simple one-pole lowpass filter
        output = np.zeros_like(audio)
        output[0] = audio[0]

        for i in range(1, len(audio)):
            output[i] = damping * output[i-1] + (1 - damping) * audio[i]

        return output

    def to_dict(self) -> dict:
        """Export reverb configuration to dictionary."""
        return {
            "decay_s": self.decay_s,
            "pre_delay_ms": self.pre_delay_ms,
            "mix": self.mix,
            "damping": self.damping,
            "room_size": self.room_size,
        }

    def from_dict(self, config: dict) -> None:
        """Load reverb configuration from dictionary."""
        self.__init__(
            sample_rate=self.sample_rate,
            decay_s=config.get("decay_s", 2.0),
            pre_delay_ms=config.get("pre_delay_ms", 20.0),
            mix=config.get("mix", 0.3),
            damping=config.get("damping", 0.5),
            room_size=config.get("room_size", 0.5),
        )


class AudioProcessingPipeline:
    """Complete audio processing pipeline combining EQ, compression, and reverb.

    This pipeline is designed to process RVC output to achieve
    the characteristic GLaDOS voice timbre.
    """

    def __init__(self, sample_rate: int = 48000):
        """Initialize the processing pipeline.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.eq = AudioEQProcessor(sample_rate)
        self.compressor = Compressor(sample_rate)
        self.reverb = Reverb(sample_rate)

        logger.info(f"Audio processing pipeline initialized at {sample_rate}Hz")

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Process audio through the complete pipeline.

        Pipeline order: EQ -> Compressor -> Reverb

        Args:
            audio: Input audio signal

        Returns:
            Processed audio signal
        """
        if len(audio) == 0:
            return audio

        # Apply EQ first to shape frequency response
        processed = self.eq.apply(audio)

        # Then compress to control dynamics
        processed = self.compressor.apply(processed)

        # Finally add reverb for space
        processed = self.reverb.apply(processed)

        return processed

    def load_glados_preset(self) -> None:
        """Load the default GLaDOS voice preset.

        This preset is designed to emulate the characteristic
        GLaDOS timbre with robotic/synthetic qualities.
        """
        # Clear existing settings
        self.eq.clear_bands()

        # GLaDOS-like EQ curve
        # High-pass to remove low rumble
        self.eq.add_band(freq=110, gain_db=0, q=0.7, filter_type='highpass')

        # Boost presence/clarity in mid-high range
        self.eq.add_band(freq=3200, gain_db=5.0, q=1.2, filter_type='peak')

        # Boost high-end for synthetic quality
        self.eq.add_band(freq=7000, gain_db=3.5, q=0.8, filter_type='highshelf')

        # Slight cut in lower mids to reduce muddiness
        self.eq.add_band(freq=400, gain_db=-2.0, q=1.0, filter_type='peak')

        # Compressor for consistent level
        self.compressor.from_dict({
            "threshold_db": -20,
            "ratio": 4.0,
            "attack_ms": 10,
            "release_ms": 100,
            "makeup_gain_db": 3,
        })

        # Reverb for the characteristic "chamber" sound
        self.reverb.from_dict({
            "decay_s": 3.0,
            "pre_delay_ms": 35,
            "mix": 0.35,
            "damping": 0.6,
            "room_size": 0.7,
        })

        logger.info("GLaDOS audio preset loaded")

    def to_dict(self) -> dict:
        """Export complete pipeline configuration to dictionary."""
        return {
            "eq": self.eq.to_dict(),
            "compressor": self.compressor.to_dict(),
            "reverb": self.reverb.to_dict(),
        }

    def from_dict(self, config: dict) -> None:
        """Load complete pipeline configuration from dictionary."""
        if "eq" in config:
            self.eq.from_dict(config["eq"])
        if "compressor" in config:
            self.compressor.from_dict(config["compressor"])
        if "reverb" in config:
            self.reverb.from_dict(config["reverb"])
