"""GLaDOS - Voice Assistant using ONNX models for speech synthesis and recognition."""

# IMPORTANT: Lazy imports to avoid argparse conflicts!
# Do NOT import modules here - fairseq and other dependencies may try to parse
# sys.argv during import, conflicting with CLI argument parsing.
# Import Glados and GladosConfig directly where needed instead.

__version__ = "0.1.0"

# Lazy imports via __getattr__ for backwards compatibility
def __getattr__(name: str):
    if name == "Glados":
        from .core.engine import Glados
        return Glados
    elif name == "GladosConfig":
        from .core.engine import GladosConfig
        return GladosConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
