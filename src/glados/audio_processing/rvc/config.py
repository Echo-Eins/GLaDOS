from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(slots=True)
class RVCConfig:
    """Runtime configuration for the RVC inference pipeline."""

    device: str
    is_half: bool
    x_pad: int = 3
    x_query: int = 10
    x_center: int = 60
    x_max: int = 65
    rms_mix_rate: float = 1.0
    protect: float = 0.33
    filter_radius: int = 3

    # Resample target for final audio. When set to -1 the model sample rate is used.
    resample_sr: int = -1

    @property
    def torch_device(self) -> str:
        return self.device


DEFAULT_CONFIG: Final[RVCConfig] = RVCConfig(device="cpu", is_half=False)
