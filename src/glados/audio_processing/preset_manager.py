"""Preset management for audio processing configurations.

Supports loading and saving EQ, Compressor, and Reverb configurations
in JSON and YAML formats.
"""

import json
from pathlib import Path
from typing import Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from loguru import logger


class PresetManager:
    """Manager for audio processing presets.

    Handles loading and saving configurations for EQ, Compressor,
    and Reverb in JSON and YAML formats.
    """

    def __init__(self, presets_dir: Path | None = None):
        """Initialize the preset manager.

        Args:
            presets_dir: Directory to store presets. If None, uses current directory.
        """
        self.presets_dir = presets_dir or Path("./presets")
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(self, name: str, config: dict[str, Any], format: str = "json") -> Path:
        """Save a preset configuration to file.

        Args:
            name: Preset name (without extension)
            config: Configuration dictionary
            format: File format ('json' or 'yaml')

        Returns:
            Path to the saved preset file

        Raises:
            ValueError: If format is unsupported or YAML is not available
        """
        if format == "yaml" and not YAML_AVAILABLE:
            raise ValueError("YAML support not available. Install PyYAML.")

        if format not in ["json", "yaml"]:
            raise ValueError(f"Unsupported format: {format}")

        preset_path = self.presets_dir / f"{name}.{format}"

        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                if format == "json":
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:  # yaml
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Preset '{name}' saved to {preset_path}")
            return preset_path

        except Exception as e:
            logger.error(f"Failed to save preset '{name}': {e}")
            raise

    def load_preset(self, name: str) -> dict[str, Any]:
        """Load a preset configuration from file.

        Automatically detects format from extension (.json or .yaml/.yml).

        Args:
            name: Preset name (with or without extension)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If preset file doesn't exist
            ValueError: If file format is unsupported
        """
        # Try with provided name first
        preset_path = self.presets_dir / name

        if not preset_path.exists():
            # Try adding extensions
            for ext in [".json", ".yaml", ".yml"]:
                test_path = self.presets_dir / f"{name}{ext}"
                if test_path.exists():
                    preset_path = test_path
                    break
            else:
                raise FileNotFoundError(f"Preset '{name}' not found in {self.presets_dir}")

        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                if preset_path.suffix == ".json":
                    config = json.load(f)
                elif preset_path.suffix in [".yaml", ".yml"]:
                    if not YAML_AVAILABLE:
                        raise ValueError("YAML support not available. Install PyYAML.")
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {preset_path.suffix}")

            logger.info(f"Preset '{name}' loaded from {preset_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load preset '{name}': {e}")
            raise

    def list_presets(self) -> list[str]:
        """List all available presets in the presets directory.

        Returns:
            List of preset names (without extensions)
        """
        presets = []

        for ext in [".json", ".yaml", ".yml"]:
            for preset_file in self.presets_dir.glob(f"*{ext}"):
                presets.append(preset_file.stem)

        return sorted(list(set(presets)))

    def delete_preset(self, name: str) -> None:
        """Delete a preset file.

        Args:
            name: Preset name (with or without extension)

        Raises:
            FileNotFoundError: If preset file doesn't exist
        """
        preset_path = self.presets_dir / name

        if not preset_path.exists():
            # Try adding extensions
            for ext in [".json", ".yaml", ".yml"]:
                test_path = self.presets_dir / f"{name}{ext}"
                if test_path.exists():
                    preset_path = test_path
                    break
            else:
                raise FileNotFoundError(f"Preset '{name}' not found in {self.presets_dir}")

        preset_path.unlink()
        logger.info(f"Preset '{name}' deleted")

    def create_default_glados_preset(self) -> Path:
        """Create and save the default GLaDOS preset.

        Returns:
            Path to the saved preset file
        """
        glados_config = {
            "eq": [
                {"type": "highpass", "frequency": 110, "gain_db": 0, "q_factor": 0.7},
                {"type": "peak", "frequency": 400, "gain_db": -2.0, "q_factor": 1.0},
                {"type": "peak", "frequency": 3200, "gain_db": 5.0, "q_factor": 1.2},
                {"type": "highshelf", "frequency": 7000, "gain_db": 3.5, "q_factor": 0.8},
            ],
            "compressor": {
                "threshold_db": -20,
                "ratio": 4.0,
                "attack_ms": 10,
                "release_ms": 100,
                "makeup_gain_db": 3,
            },
            "reverb": {
                "decay_s": 3.0,
                "pre_delay_ms": 35,
                "mix": 0.35,
                "damping": 0.6,
                "room_size": 0.7,
            }
        }

        return self.save_preset("glados_default", glados_config, format="json")

    def create_example_presets(self) -> None:
        """Create several example presets for different voice characteristics."""
        # GLaDOS Classic
        self.create_default_glados_preset()

        # GLaDOS Subtle (less processed)
        subtle_config = {
            "eq": [
                {"type": "highpass", "frequency": 100, "gain_db": 0, "q_factor": 0.7},
                {"type": "peak", "frequency": 3000, "gain_db": 2.5, "q_factor": 1.0},
            ],
            "compressor": {
                "threshold_db": -25,
                "ratio": 2.5,
                "attack_ms": 15,
                "release_ms": 150,
                "makeup_gain_db": 2,
            },
            "reverb": {
                "decay_s": 1.5,
                "pre_delay_ms": 20,
                "mix": 0.2,
                "damping": 0.5,
                "room_size": 0.5,
            }
        }
        self.save_preset("glados_subtle", subtle_config)

        # GLaDOS Enhanced (more dramatic)
        enhanced_config = {
            "eq": [
                {"type": "highpass", "frequency": 120, "gain_db": 0, "q_factor": 0.8},
                {"type": "peak", "frequency": 300, "gain_db": -3.0, "q_factor": 1.2},
                {"type": "peak", "frequency": 3500, "gain_db": 7.0, "q_factor": 1.5},
                {"type": "highshelf", "frequency": 8000, "gain_db": 5.0, "q_factor": 0.7},
            ],
            "compressor": {
                "threshold_db": -18,
                "ratio": 6.0,
                "attack_ms": 5,
                "release_ms": 80,
                "makeup_gain_db": 5,
            },
            "reverb": {
                "decay_s": 4.5,
                "pre_delay_ms": 50,
                "mix": 0.45,
                "damping": 0.7,
                "room_size": 0.85,
            }
        }
        self.save_preset("glados_enhanced", enhanced_config)

        logger.info("Example presets created successfully")
