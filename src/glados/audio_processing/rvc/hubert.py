from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from fairseq import checkpoint_utils


def load_hubert_model(checkpoint_path: Path, device: torch.device, *, is_half: bool) -> torch.nn.Module:
    """Load a HuBERT model used by the RVC pipeline."""

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint_path)], suffix="")
    hubert_model = models[0]
    hubert_model.to(device)
    hubert_model.eval()
    if is_half:
        hubert_model.half()
    else:
        hubert_model.float()
    return hubert_model


def resolve_hubert_checkpoint(default_root: Path) -> Optional[Path]:
    """Return the path to the HuBERT checkpoint if it exists."""

    potential = default_root / "hubert_base.pt"
    if potential.exists():
        return potential
    return None
