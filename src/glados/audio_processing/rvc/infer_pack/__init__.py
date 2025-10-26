"""Subset of RVC inference modules vendored from the official RVC WebUI project."""

from . import attentions, commons, modules
from .models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

__all__ = [
    "attentions",
    "commons",
    "modules",
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs768NSFsid_nono",
]
