"""Utilities for collecting GPU telemetry information."""

from __future__ import annotations

from contextlib import suppress

from loguru import logger

try:  # pragma: no cover - optional dependency detection
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - import may fail for many reasons
    pynvml = None  # type: ignore


def get_gpu_load_percentage() -> float | None:
    """Return the average GPU load percentage across available devices.

    The function tries to use NVIDIA's NVML bindings provided by ``pynvml`` or
    ``nvidia-ml-py``. When NVML isn't available or no GPUs are detected the
    function returns ``None``. Any NVML errors are logged at the debug level so
    that the UI can safely fall back to a disabled indicator.
    """

    if pynvml is None:
        logger.debug("pynvml is not available; skipping GPU telemetry")
        return None

    try:
        pynvml.nvmlInit()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - NVML specific errors
        logger.debug("Failed to initialise NVML: {}", exc)
        return None

    try:
        count = pynvml.nvmlDeviceGetCount()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to query NVML device count: {}", exc)
        with suppress(Exception):  # pragma: no cover - best effort shutdown
            pynvml.nvmlShutdown()  # type: ignore[attr-defined]
        return None

    if not isinstance(count, int) or count <= 0:
        with suppress(Exception):  # pragma: no cover
            pynvml.nvmlShutdown()  # type: ignore[attr-defined]
        return None

    total = 0.0
    active_devices = 0

    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # type: ignore[attr-defined]
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - NVML specific errors
            logger.debug("Failed to query NVML utilisation for device {}: {}", index, exc)
            continue

        gpu_util = getattr(utilization, "gpu", None)
        if gpu_util is None:
            continue
        try:
            gpu_util_float = float(gpu_util)
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            continue

        total += gpu_util_float
        active_devices += 1

    with suppress(Exception):  # pragma: no cover
        pynvml.nvmlShutdown()  # type: ignore[attr-defined]

    if active_devices == 0:
        return None

    return max(0.0, min(100.0, total / active_devices))
