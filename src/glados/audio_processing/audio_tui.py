"""Text User Interface (TUI) for audio processing configuration.

Interactive TUI for real-time adjustment of EQ, Compressor, and Reverb parameters.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Button, Label, Input
from textual.reactive import reactive
from loguru import logger

from .audio_processor import AudioProcessingPipeline
from .preset_manager import PresetManager


class EQBandWidget(Static):
    """Widget for controlling a single EQ band."""

    def __init__(self, band_index: int, band_config: dict):
        super().__init__()
        self.band_index = band_index
        self.config = band_config

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Vertical():
            yield Label(f"Band {self.band_index + 1}: {self.config['type']}")
            yield Label(f"  Frequency: {self.config['frequency']}Hz")
            yield Label(f"  Gain: {self.config['gain_db']}dB")
            yield Label(f"  Q: {self.config['q_factor']}")


class AudioProcessorTUI(App):
    """TUI application for audio processor configuration."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    .section {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    .section-title {
        color: $accent;
        text-style: bold;
    }

    .param-label {
        width: 20;
    }

    .param-value {
        width: 15;
    }

    Button {
        margin: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "save_preset", "Save Preset"),
        ("l", "load_preset", "Load Preset"),
        ("r", "reset_default", "Reset to Default"),
    ]

    current_preset = reactive("glados_default")

    def __init__(self, pipeline: AudioProcessingPipeline | None = None):
        """Initialize TUI.

        Args:
            pipeline: Audio processing pipeline to configure
        """
        super().__init__()
        self.pipeline = pipeline or AudioProcessingPipeline()
        self.preset_manager = PresetManager()

    def compose(self) -> ComposeResult:
        """Create child widgets for app."""
        yield Header(show_clock=True)

        with VerticalScroll(id="main-container"):
            # EQ Section
            with Container(classes="section"):
                yield Label("Equalizer", classes="section-title")
                yield Label(f"Bands: {len(self.pipeline.eq.bands)}")

                for i, band in enumerate(self.pipeline.eq.bands):
                    yield EQBandWidget(i, {
                        "type": band.filter_type,
                        "frequency": band.frequency,
                        "gain_db": band.gain_db,
                        "q_factor": band.q_factor,
                    })

            # Compressor Section
            with Container(classes="section"):
                yield Label("Compressor", classes="section-title")
                comp = self.pipeline.compressor

                with Horizontal():
                    yield Label("Threshold:", classes="param-label")
                    yield Label(f"{comp.threshold_db} dB", classes="param-value")

                with Horizontal():
                    yield Label("Ratio:", classes="param-label")
                    yield Label(f"{comp.ratio}:1", classes="param-value")

                with Horizontal():
                    yield Label("Attack:", classes="param-label")
                    yield Label(f"{comp.attack_ms} ms", classes="param-value")

                with Horizontal():
                    yield Label("Release:", classes="param-label")
                    yield Label(f"{comp.release_ms} ms", classes="param-value")

                with Horizontal():
                    yield Label("Makeup Gain:", classes="param-label")
                    yield Label(f"{comp.makeup_gain_db} dB", classes="param-value")

            # Reverb Section
            with Container(classes="section"):
                yield Label("Reverb", classes="section-title")
                rev = self.pipeline.reverb

                with Horizontal():
                    yield Label("Decay Time:", classes="param-label")
                    yield Label(f"{rev.decay_s} s", classes="param-value")

                with Horizontal():
                    yield Label("Pre-delay:", classes="param-label")
                    yield Label(f"{rev.pre_delay_ms} ms", classes="param-value")

                with Horizontal():
                    yield Label("Mix:", classes="param-label")
                    yield Label(f"{rev.mix * 100:.1f}%", classes="param-value")

                with Horizontal():
                    yield Label("Damping:", classes="param-label")
                    yield Label(f"{rev.damping:.2f}", classes="param-value")

                with Horizontal():
                    yield Label("Room Size:", classes="param-label")
                    yield Label(f"{rev.room_size:.2f}", classes="param-value")

            # Preset Management Section
            with Container(classes="section"):
                yield Label("Preset Management", classes="section-title")

                with Horizontal():
                    yield Label(f"Current: {self.current_preset}")

                with Horizontal():
                    yield Button("Save Preset", id="save-btn", variant="primary")
                    yield Button("Load Preset", id="load-btn", variant="success")
                    yield Button("Reset to Default", id="reset-btn", variant="warning")

                yield Label("Available Presets:")
                presets = self.preset_manager.list_presets()
                for preset in presets:
                    yield Label(f"  - {preset}")

        yield Footer()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_save_preset(self) -> None:
        """Save current configuration as preset."""
        try:
            config = self.pipeline.to_dict()
            preset_name = f"custom_{self.current_preset}"
            self.preset_manager.save_preset(preset_name, config)
            self.notify(f"Preset saved: {preset_name}", severity="information")
            logger.info(f"Preset saved: {preset_name}")
        except Exception as e:
            self.notify(f"Failed to save preset: {e}", severity="error")
            logger.error(f"Failed to save preset: {e}")

    def action_load_preset(self) -> None:
        """Load a preset configuration."""
        try:
            config = self.preset_manager.load_preset(self.current_preset)
            self.pipeline.from_dict(config)
            self.notify(f"Preset loaded: {self.current_preset}", severity="information")
            logger.info(f"Preset loaded: {self.current_preset}")
            self.refresh()
        except Exception as e:
            self.notify(f"Failed to load preset: {e}", severity="error")
            logger.error(f"Failed to load preset: {e}")

    def action_reset_default(self) -> None:
        """Reset to default GLaDOS preset."""
        try:
            self.pipeline.load_glados_preset()
            self.current_preset = "glados_default"
            self.notify("Reset to GLaDOS default preset", severity="information")
            logger.info("Reset to GLaDOS default preset")
            self.refresh()
        except Exception as e:
            self.notify(f"Failed to reset: {e}", severity="error")
            logger.error(f"Failed to reset: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "save-btn":
            self.action_save_preset()
        elif button_id == "load-btn":
            self.action_load_preset()
        elif button_id == "reset-btn":
            self.action_reset_default()


def run_audio_processor_tui(pipeline: AudioProcessingPipeline | None = None) -> None:
    """Run the audio processor TUI application.

    Args:
        pipeline: Audio processing pipeline to configure
    """
    app = AudioProcessorTUI(pipeline)
    app.run()


if __name__ == "__main__":
    # Run standalone
    run_audio_processor_tui()
