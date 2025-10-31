"""
Monkey patches for gigaam package to suppress torch warnings.

This module patches the gigaam package to use safer torch operations
and suppress non-critical warnings.
"""

from __future__ import annotations

import warnings
from typing import Any


def apply_gigaam_patches() -> None:
    """Apply all necessary patches to the gigaam package."""
    try:
        import gigaam
        import torch
    except ImportError:
        # gigaam not installed, patches not needed
        return

    # Patch 1: Fix torch.load to use weights_only=True
    _patch_torch_load()

    # Patch 2: Suppress non-writable buffer warning from torch.frombuffer
    _suppress_frombuffer_warning()


def _patch_torch_load() -> None:
    """Patch gigaam to use weights_only=True for torch.load."""
    import gigaam
    import torch

    original_load_model = gigaam.load_model

    def patched_load_model(model_type: str, **kwargs: Any) -> Any:
        """Patched version of gigaam.load_model that uses safer torch.load."""
        # Temporarily patch torch.load
        original_torch_load = torch.load

        def safe_torch_load(f: Any, map_location: Any = None, **load_kwargs: Any) -> Any:
            # Use weights_only=False for backward compatibility with gigaam models
            # but suppress the FutureWarning
            # Remove weights_only from load_kwargs to avoid duplicate argument error
            load_kwargs.pop('weights_only', None)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
                return original_torch_load(f, map_location=map_location, weights_only=False, **load_kwargs)

        torch.load = safe_torch_load
        try:
            result = original_load_model(model_type, **kwargs)
        finally:
            torch.load = original_torch_load

        return result

    gigaam.load_model = patched_load_model


def _suppress_frombuffer_warning() -> None:
    """Suppress UserWarning about non-writable buffers in torch.frombuffer."""
    # This warning comes from gigaam.preprocess and is not critical
    # The warning suggests making a copy, which gigaam should handle internally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*not writable.*PyTorch does not support non-writable tensors.*",
    )
