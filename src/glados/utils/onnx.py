"""Utilities for working with ONNX Runtime providers.

This module centralises provider selection so every component that uses
ONNX Runtime benefits from the same CUDA-aware logic and fallbacks.  It
ensures that GPU accelerators are prioritised when available while still
remaining compatible with CPU-only environments.
"""

from __future__ import annotations

from typing import Iterable

import onnxruntime as ort  # type: ignore

_FALLBACK_PROVIDER = "CPUExecutionProvider"
_EXCLUDED_PROVIDERS = {
    "TensorrtExecutionProvider",
    "CoreMLExecutionProvider",
}
_DEFAULT_PREFERRED = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
)


def _normalise_preferred(preferred: Iterable[str] | None) -> tuple[str, ...]:
    if preferred is None:
        return _DEFAULT_PREFERRED
    return tuple(preferred)


def get_available_providers(preferred: Iterable[str] | None = None) -> list[str]:
    """Return a list of ONNX Runtime providers prioritising accelerators.

    Parameters
    ----------
    preferred:
        A collection of provider names ordered by priority.  Providers not
        present on the current system are ignored.  When *preferred* is not
        supplied we default to preferring CUDA and DirectML providers.

    Returns
    -------
    list[str]
        A provider list suitable for passing to :class:`onnxruntime.InferenceSession`.
        The list always includes a CPU fallback to keep compatibility with
        systems where accelerators are partially available or fail to
        initialise.
    """

    if not hasattr(ort, "get_available_providers"):
        return [_FALLBACK_PROVIDER]

    available = [p for p in ort.get_available_providers() if p not in _EXCLUDED_PROVIDERS]

    ordered: list[str] = []
    for provider in _normalise_preferred(preferred):
        if provider in available:
            ordered.append(provider)
            available.remove(provider)

    if _FALLBACK_PROVIDER in available:
        available.remove(_FALLBACK_PROVIDER)
        ordered.append(_FALLBACK_PROVIDER)
    elif _FALLBACK_PROVIDER not in ordered:
        ordered.append(_FALLBACK_PROVIDER)

    ordered.extend(p for p in available if p not in ordered)

    return ordered