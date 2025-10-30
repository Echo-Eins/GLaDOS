"""Telemetry helpers for monitoring system resources."""

from .gpu import get_gpu_load_percentage

__all__ = ["get_gpu_load_percentage"]
