from collections.abc import Iterator
from pathlib import Path
import random
import sys
from typing import ClassVar, cast
import yaml
import queue
import itertools
from collections import deque
from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Digits, Input
from textual.widgets import Footer, Header, Label, Log, ProgressBar, RichLog, Static
from textual.worker import Worker, WorkerState
import numpy as np
from glados.core.engine import Glados, GladosConfig
from glados.glados_ui.text_resources import aperture, help_text, login_text, recipe
from glados.telemetry import get_gpu_load_percentage

# Custom Widgets

class SpectrumWidget(Static):
    """Visualize FFT band energies as animated columns."""

    DEFAULT_CSS = """
    SpectrumWidget {
        color: $primary;
        padding: 0 1;
        min-height: 10;
    }
    """

    def __init__(
        self,
        data_queue: queue.Queue[np.ndarray] | None = None,
        *,
        num_bands: int = 16,
        history: int = 12,
        height: int = 10,
        update_interval: float = 0.1,
        decay: float = 0.92,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._queue = data_queue
        self._num_bands = num_bands
        self._history = deque(maxlen=max(1, history))
        self._height = max(3, height)
        self._update_interval = update_interval
        self._decay = decay
        self._blank_text = Text("\n".join([" " * (self._num_bands * 2 - 1)] * self._height))

    def set_data_queue(self, data_queue: queue.Queue[np.ndarray] | None) -> None:
        """Attach a queue that supplies spectrum frames."""

        self._queue = data_queue

    def on_mount(self) -> None:
        self.set_interval(self._update_interval, self._refresh)

    def _refresh(self) -> None:
        updated = self._ingest_frames()

        if not self._history:
            self.update(self._blank_text)
            return

        if not updated:
            decayed = np.copy(self._history[-1]) * self._decay
            np.clip(decayed, 0.0, None, out=decayed)
            self._history.append(decayed)

        self._render()

    def _ingest_frames(self) -> bool:
        if self._queue is None:
            return False

        received = False
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                break

            arr = np.asarray(frame, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.size != self._num_bands:
                arr = self._resample(arr, self._num_bands)

            self._history.append(arr)
            received = True

        return received

    def _resample(self, values: np.ndarray, target: int) -> np.ndarray:
        x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=target, endpoint=True)
        resampled = np.interp(x_new, x_old, values)
        return resampled.astype(np.float32)

    def _render(self) -> None:
        stack = np.vstack(self._history)
        averaged = stack.mean(axis=0)
        max_value = float(np.max(averaged)) if averaged.size else 0.0
        if max_value <= 0:
            normalized = np.zeros(self._num_bands, dtype=np.float32)
        else:
            normalized = np.clip(averaged / max_value, 0.0, 1.0)

        levels = np.round(normalized * self._height).astype(int)
        rows: list[str] = []
        for level in range(self._height, 0, -1):
            row_segments: list[str] = []
            for idx, band_level in enumerate(levels):
                if band_level >= level:
                    color = self._color_for_value(normalized[idx])
                    row_segments.append(f"[{color}]█[/]")
                else:
                    row_segments.append(" ")
            rows.append(" ".join(row_segments))

        baseline = "[dim]" + "─ " * self._num_bands + "[/dim]"
        rows.append(baseline.rstrip())
        self.update(Text("\n".join(rows)))

    @staticmethod
    def _color_for_value(value: float) -> str:
        if value < 0.25:
            return "#1f77b4"
        if value < 0.5:
            return "#2ca02c"
        if value < 0.75:
            return "#ffb000"
        return "#ff4f4f"


class Printer(RichLog):
    """A subclass of textual's RichLog which captures and displays all print calls."""

    def on_mount(self) -> None:
        self.wrap = True
        self.markup = True
        self.begin_capture_print()

    def on_print(self, event: events.Print) -> None:
        if (text := event.text) != "\n":
            self.write(text.rstrip().replace("DEBUG", "[red]DEBUG[/]"))


class ScrollingBlocks(Log):
    """A widget for displaying random scrolling blocks."""

    BLOCKS = "⚊⚌☰𝌆䷀"
    DEFAULT_CSS = """
    ScrollingBlocks {
        scrollbar_size: 0 0;
        overflow-x: hidden;
    }"""

    def _animate_blocks(self) -> None:
        # Create a string of blocks of the right length, allowing
        # for border and padding
        """
        Generates and writes a line of random block characters to the log.

        This method creates a string of random block characters with a length adjusted
        to fit the current widget width, accounting for border and padding. Each block
        is randomly selected from the predefined `BLOCKS` attribute.

        The generated line is written to the log using `write_line()`, creating a
        visually dynamic scrolling effect of random block characters.

        Parameters:
            None

        Returns:
            None
        """
        # Ensure width calculation doesn't go negative if self.size.width is small
        num_blocks_to_generate = max(0, self.size.width - 8)
        random_blocks = " ".join(random.choice(self.BLOCKS) for _ in range(num_blocks_to_generate))
        self.write_line(f"{random_blocks}")

    def on_show(self) -> None:
        """
        Set up an interval timer to periodically animate scrolling blocks.

        This method is called when the widget becomes visible, initiating a recurring animation
        that calls the `_animate_blocks` method at a fixed time interval of 0.18 seconds.

        The interval timer ensures continuous block animation while the widget is displayed.
        """
        self.set_interval(0.18, self._animate_blocks)


class GPULoadWidget(Static):
    """Display the current GPU utilisation as a progress bar."""

    DEFAULT_CSS = """
    GPULoadWidget {
        layout: vertical;
        padding: 0 1;
        width: 100%;
    }

    GPULoadWidget > Label {
        text-align: center;
    }
    """

    def __init__(self, *, update_interval: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._update_interval = update_interval
        self._label: Label | None = None
        self._progress: ProgressBar | None = None

    def compose(self) -> ComposeResult:
        self._label = Label("GPU Load: N/A", id="gpu_load_label")
        self._progress = ProgressBar(
            id="gpu_progress_bar",
            total=100,
            show_eta=False,
            show_percentage=False,
        )
        self._progress.update(progress=0)
        yield self._label
        yield self._progress

    def on_mount(self) -> None:
        self.set_interval(self._update_interval, self._refresh)

    def _refresh(self) -> None:
        load = get_gpu_load_percentage()
        if load is None:
            self._set_no_gpu()
            return

        load = max(0.0, min(100.0, load))
        self._set_gpu_load(load)

    def _set_no_gpu(self) -> None:
        if self._label is None or self._progress is None:
            return
        self._label.update("GPU Load: N/A")
        # Textual doesn't recognise colour names like "grey62", so use hex codes
        self._label.styles.color = "#9e9e9e"
        self._progress.update(progress=0)
        self._progress.bar_style = "#4d4d4d"

    def _set_gpu_load(self, load: float) -> None:
        if self._label is None or self._progress is None:
            return

        colour = self._colour_for_load(load)
        self._label.update(f"GPU Load: {int(round(load))}%")
        self._label.styles.color = colour
        self._progress.bar_style = colour
        self._progress.update(progress=load)

    @staticmethod
    def _colour_for_load(load: float) -> str:
        if load < 40:
            return "#2ca02c"
        if load < 75:
            return "#ffb000"
        return "#ff4f4f"

class PipelineStatusWidget(Static):
    """Display the high-level state of the voice pipeline."""

    DEFAULT_CSS = """
    PipelineStatusWidget {
        layout: vertical;
        padding: 1 1;
        border: solid $primary 60%;
        background: $background 85%;
        color: $foreground 80%;
        gap: 1;
    }

    PipelineStatusWidget > .status-title {
        text-style: bold;
        color: $primary;
    }
    """

    SPINNER_FRAMES: ClassVar[tuple[str, ...]] = (
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    )

    def __init__(
        self,
        glados: Glados | None = None,
        *,
        update_interval: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._glados = glados
        self._status_queue: queue.Queue[tuple[str, bool]] | None = None
        self._update_interval = update_interval
        self._spinner_cycle = itertools.cycle(self.SPINNER_FRAMES)
        self._current_frame = next(self._spinner_cycle)
        self._listen_active = True
        self._think_active = False
        self._speak_active = False
        self._last_render: tuple[bool, bool, bool] | None = None
        self._listen_label: Label | None = None
        self._think_label: Label | None = None
        self._speak_label: Label | None = None

    def compose(self) -> ComposeResult:
        yield Label("PIPELINE", classes="status-title")
        self._listen_label = Label()
        self._think_label = Label()
        self._speak_label = Label()
        yield self._listen_label
        yield self._think_label
        yield self._speak_label

    def on_mount(self) -> None:
        self.set_interval(self._update_interval, self._refresh)
        self._render_state(force=True)

    def set_glados(self, glados: Glados | None) -> None:
        """Bind the widget to a Glados engine instance."""

        self._glados = glados
        if glados is not None:
            status_queue = getattr(glados, "pipeline_status_queue", None)
            if isinstance(status_queue, queue.Queue):
                self._status_queue = status_queue
            else:
                self._status_queue = None
                self._think_active = False
        else:
            self._status_queue = None
            self._think_active = False
            self._speak_active = False
            self._listen_active = True
        self._sample_events()
        self._render_state(force=True)

    def _refresh(self) -> None:
        self._ingest_status_updates()
        self._sample_events()
        self._render_state()

    def _ingest_status_updates(self) -> None:
        if self._status_queue is None:
            return

        while True:
            try:
                state, active = self._status_queue.get_nowait()
            except queue.Empty:
                break

            if state == "think":
                self._think_active = active

    def _sample_events(self) -> None:
        glados = self._glados
        if glados is None:
            self._listen_active = True
            self._speak_active = False
            return

        speak_active = glados.currently_speaking_event.is_set()
        processing_active = glados.processing_active_event.is_set()

        self._listen_active = not processing_active and not speak_active and not self._think_active
        self._speak_active = speak_active

    def _render_state(self, *, force: bool = False) -> None:
        state = (self._listen_active, self._think_active, self._speak_active)
        if not force and state == self._last_render:
            return

        if self._listen_label is None or self._think_label is None or self._speak_label is None:
            return

        self._current_frame = next(self._spinner_cycle)
        self._listen_label.update(self._format_line("LISTEN", self._listen_active))
        self._think_label.update(self._format_line("THINK", self._think_active))
        self._speak_label.update(self._format_line("SPEAK", self._speak_active))
        self._last_render = state

    def _format_line(self, title: str, active: bool) -> Text:
        if active:
            markup = f"[bold {self._active_colour(title)}]{title:<6}[/] {self._current_frame}"
        else:
            markup = f"[dim]{title:<6}[/] ·"
        return Text.from_markup(markup)

    @staticmethod
    def _active_colour(title: str) -> str:
        palette = {
            "LISTEN": "#2ca02c",
            "THINK": "#1f77b4",
            "SPEAK": "#ff4f4f",
        }
        return palette.get(title, "#ffb000")


class Typewriter(Static):
    """A widget which displays text a character at a time."""

    def __init__(
        self,
        text: str = "_",
        id: str | None = None,  # Consistent with typical Textual widget `id` parameter
        speed: float = 0.01,  # time between each character
        repeat: bool = False,  # whether to start again at the end
        # Static widget parameters
        content: str = "",
        expand: bool = False,
        shrink: bool = False,
        markup: bool = True,
        name: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        # Initialize our custom attributes first
        self._text = text
        self.__id_for_child = id  # Store id specifically for the child VerticalScroll
        self._speed = speed
        self._repeat = repeat
        # Flag to determine if we should use Rich markup
        self._use_markup = True
        # Check if text contains special Rich markup characters
        if "[" in text or "]" in text:
            # If there are brackets in the text, disable markup to avoid conflicts
            self._use_markup = False

        # Call parent constructor with proper parameters
        super().__init__(
            content,
            expand=expand,
            shrink=shrink,
            markup=markup,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

    def compose(self) -> ComposeResult:
        self._static = Static(markup=self._use_markup)
        self._vertical_scroll = VerticalScroll(self._static, id=self.__id_for_child)
        yield self._vertical_scroll

    def _get_iterator(self) -> Iterator[str]:
        """
        Create an iterator that returns progressively longer substrings of the text,
        with a cursor at the end.

        If markup is enabled, uses a blinking cursor with Rich markup.
        If markup is disabled (due to brackets in the text), uses a plain underscore.
        """
        if self._use_markup:
            # Use Rich markup for the blinking cursor if markup is enabled
            return (self._text[:i] + "[blink]_[/blink]" for i in range(len(self._text) + 1))
        else:
            # Use a simple underscore cursor if markup is disabled
            return (self._text[:i] + "_" for i in range(len(self._text) + 1))

    def on_mount(self) -> None:
        self._iter_text = self._get_iterator()
        self.set_interval(self._speed, self._display_next_char)

    def _display_next_char(self) -> None:
        """Get and display the next character."""
        try:
            # Scroll down first, then update. This feels more natural for a typewriter.
            if not self._vertical_scroll.is_vertical_scroll_end:
                self._vertical_scroll.scroll_down()
            self._static.update(next(self._iter_text))
        except StopIteration:
            if self._repeat:
                self._iter_text = self._get_iterator()
            # else:
            # Optional: If not repeating, remove the cursor or show final text without cursor.
            # For example: self._static.update(self._text)


# Screens
class SplashScreen(Screen[None]):
    """Splash screen shown on startup."""

    # Ensure this path is correct relative to your project structure/runtime directory
    # Using a try-except block for robustness if the file is missing
    try:
        with open(Path("src/glados/glados_ui/images/splash.ansi"), encoding="utf-8") as f:
            SPLASH_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Splash screen ANSI art file not found. Using placeholder.")
        SPLASH_ANSI = Text.from_markup("[bold red]Splash ANSI Art Missing[/bold red]")

    def __init__(self) -> None:
        super().__init__()
        self._password_modal_open = False

    def compose(self) -> ComposeResult:
        """
        Compose the layout for the splash screen.

        This method defines the visual composition of the SplashScreen, creating a container
        with a logo, a banner, and a typewriter-style login text.

        Returns:
            ComposeResult: A generator yielding the screen's UI components, including:
                - A container with a static ANSI logo
                - A label displaying the aperture text
                - A typewriter-animated login text with a slow character reveal speed
        """
        with Container(id="splash_logo_container"):
            yield Static(self.SPLASH_ANSI, id="splash_logo")
            yield Label(aperture, id="banner")
        yield Typewriter(login_text, id="login_text", speed=0.0075)

    def on_mount(self) -> None:
        """
        Automatically scroll the widget to its bottom at regular intervals.

        This method sets up a periodic timer to ensure the widget always displays
        the most recent content by scrolling to the end. The scrolling occurs
        every 0.5 seconds, providing a smooth and continuous view of the latest information.

        Args:
            None

        Returns:
            None
        """
        self.set_interval(0.5, self.scroll_end)

    def on_key(self, event: events.Key) -> None:
        """Prompt for the password when any key is pressed."""
        self._prompt_for_password()

    def _prompt_for_password(self) -> None:
        app = cast(GladosUI, self.app)
        if self._password_modal_open:
            return
        if not app.login_password:
            self._start_glados()
            return
        self._password_modal_open = True
        app.push_screen(PasswordModal(self))

    def _start_glados(self) -> None:
        app = cast(GladosUI, self.app)
        if app.glados_engine_instance:
            app.glados_engine_instance.play_announcement()
            app.start_glados()
            self.dismiss()
        else:
            app.notify("AI engine is not ready yet.", severity="warning")

        def handle_login_success(self) -> None:
            self._start_glados()

        def on_password_modal_closed(self) -> None:
            self._password_modal_open = False

        class PasswordModal(ModalScreen[None]):
            """Modal screen for entering the login password."""

            def __init__(self, splash_screen: "SplashScreen") -> None:
                super().__init__()
                self._splash_screen = splash_screen

            def compose(self) -> ComposeResult:
                with Container(id="password_dialog"):
                    with Vertical(id="password_dialog_content"):
                        yield Label("ENTER PASSWORD", id="password_prompt")
                        yield Input(
                            password=True,
                            placeholder="••••••",
                            id="password_input",
                        )

            def on_mount(self) -> None:
                password_input = self.query_one("#password_input", Input)
                password_input.focus()

            def on_input_submitted(self, event: Input.Submitted) -> None:
                app = cast(GladosUI, self.app)
                entered_password = event.value
                event.input.value = ""
                expected_password = app.login_password or ""

                if entered_password == expected_password:
                    self._splash_screen.handle_login_success()
                    self.dismiss()
                else:
                    app.notify("Неверный пароль", severity="error")
                    event.input.focus()

            def on_key(self, event: events.Key) -> None:
                if event.key == "escape":
                    self.dismiss()
                    event.stop()

            def dismiss(self, result: object | None = None) -> None:
                self._splash_screen.on_password_modal_closed()
                super().dismiss(result)


class HelpScreen(ModalScreen[None]):
    """The help screen. Possibly not that helpful."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Close screen")
    ]

    TITLE = "Help"

    def compose(self) -> ComposeResult:
        """
        Compose the help screen's layout by creating a container with a typewriter widget.

        This method generates the visual composition of the help screen, wrapping the help text
        in a Typewriter widget for an animated text display within a Container.

        Returns:
            ComposeResult: A generator yielding the composed help screen container with animated text.
        """
        yield Container(Typewriter(help_text, id="help_text"), id="help_dialog")

    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = self.TITLE
        # Consistent use of explicit closing tag for blink
        dialog.border_subtitle = "[blink]Press Esc key to continue[/blink]"


# The App
class GladosUI(App[None]):
    """The main app class for the GlaDOS ui."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(
            key="question_mark",
            action="help",
            description="Help",
            key_display="?",
        ),
    ]
    CSS_PATH = "glados_ui/glados.tcss"

    ENABLE_COMMAND_PALETTE = False

    TITLE = "GlaDOS v 1.09"

    SUB_TITLE = "(c) 1982 Aperture Science, Inc."

    try:
        with open(Path("src/glados/glados_ui/images/logo.ansi"), encoding="utf-8") as f:
            LOGO_ANSI = Text.from_ansi(f.read(), no_wrap=True, end="")
    except FileNotFoundError:
        logger.error("Logo ANSI art file not found. Using placeholder.")
        LOGO_ANSI = Text.from_markup("[bold red]Logo ANSI Art Missing[/bold red]")

    glados_engine_instance: Glados | None = None
    glados_worker: object | None = None
    instantiation_worker: Worker[None] | None = None

    def __init__(
        self,
        config_path: str | Path = "configs/glados_config.yaml",
        language: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the GladosUI application.

        Parameters:
            config_path: Path to the Glados configuration YAML file
            language: Optional language override ('en' or 'ru')
            **kwargs: Additional arguments passed to App.__init__
        """
        super().__init__(**kwargs)
        self.config_path = Path(config_path)
        self.language = language
        self.spectrum_widget: SpectrumWidget | None = None
        self.pipeline_status_widget: PipelineStatusWidget | None = None
        self.gpu_widget: GPULoadWidget | None = None
        self.login_password: str | None = None
        self._load_tui_settings()

    def compose(self) -> ComposeResult:
        """
        Compose the user interface layout for the GladosUI application.

        This method generates the primary UI components, including a header, body with log and utility areas,
        a footer, and additional decorative blocks. The layout is structured to display:
        - A header with a clock
        - A body containing:
          - A log area (Printer widget)
          - A utility area with a typewriter displaying a recipe
        - A footer
        - Additional decorative elements like scrolling blocks, text digits, and a logo

        Returns:
            ComposeResult: A generator yielding Textual UI components for rendering
        """
        # It would be nice to have the date in the header, but see:
        # https://github.com/Textualize/textual/issues/4666
        yield Header(show_clock=True)

        with Container(id="body"):
            with Horizontal(id="main_columns"):
                yield Printer(id="log_area")
                with Vertical(id="utility_area"):
                    self.pipeline_status_widget = PipelineStatusWidget(id="pipeline_status")
                    yield self.pipeline_status_widget
                    self.gpu_widget = GPULoadWidget(id="gpu_load")
                    yield self.gpu_widget
                    typewriter = Typewriter(recipe, id="recipe", speed=0.01, repeat=True)
                    yield typewriter

        yield Footer()

        with Container(id="block_container", classes="fadeable"):
            yield ScrollingBlocks(id="scrolling_block", classes="block")
            with Vertical(id="spectrum_block", classes="block"):
                self.spectrum_widget = SpectrumWidget(id="spectrum_widget")
                yield self.spectrum_widget
            yield Label(self.LOGO_ANSI, id="logo_block", classes="block")

    def on_load(self) -> None:
        """
        Configure logging settings when the application starts.

        This method is called during the application initialization, before the
        terminal enters app mode. It sets up a custom logging format and ensures
        that all log messages are printed.

        Key actions:
            - Removes any existing log handlers
            - Adds a new log handler that prints messages with a detailed, formatted output
            - Enables capturing of log text by the main log widget

        The log format includes:
            - Timestamp (YYYY-MM-DD HH:mm:ss.SSS)
            - Log level (padded to 8 characters)
            - Module name
            - Function name
            - Line number
            - Log message
        """
        # Cause logger to print all log text. Printed text can then be  captured
        # by the main_log widget

        logger.remove()
        fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}"

        self.instantiation_worker = None  # Reset the instantiation worker reference
        self.start_instantiation()

        logger.add(print, format=fmt, level="SUCCESS")  # Changed to DEBUG for more verbose logging during dev

    def on_mount(self) -> None:
        """
        Mount the application and display the initial splash screen.

        This method is called when the application is first mounted, pushing the SplashScreen
        onto the screen stack to provide a welcome or loading experience for the user before
        transitioning to the main application interface.

        Returns:
            None: Does not return any value, simply initializes the splash screen.
        """
        # Display the splash screen for a few moments
        self.push_screen(SplashScreen())
        self.notify("Loading AI engine...", title="GLaDOS", timeout=6)

    def on_unmount(self) -> None:
        """
        Called when the app is quitting.

        Makes sure that the GLaDOS engine is gracefully shut down.
        """
        logger.info("Quit action initiated in TUI.")
        if hasattr(self, "glados_engine_instance") and self.glados_engine_instance is not None:
            logger.info("Signalling GLaDOS engine to stop...")
            self.glados_engine_instance.shutdown_event.set()

    def action_help(self) -> None:
        """Someone pressed the help key!."""
        self.push_screen(HelpScreen(id="help_screen"))

    # def on_key(self, event: events.Key) -> None:
    #     """Useful for debugging via key presses."""
    #     logger.success(f"Key pressed: {self.glados_worker}")

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        """Handle messages from workers."""

        if message.state == WorkerState.SUCCESS:
            self.notify("AI Engine operational", title="GLaDOS", timeout=2)
        elif message.state == WorkerState.ERROR:
            self.notify("Instantiation failed!", severity="error")

        self.instantiation_worker = None  # Clear the worker reference

    def start_glados(self) -> None:
        """
        Start the GLaDOS worker thread in the background.

        This method initializes a worker thread to run the GLaDOS module's start function.
        The worker is run exclusively and in a separate thread to prevent blocking the main application.

        Notes:
            - Uses `run_worker` to create a non-blocking background task
            - Sets the worker as an instance attribute for potential later reference
            - The `exclusive=True` parameter ensures only one instance of this worker runs at a time
        """
        try:
            # Run in a thread to avoid blocking the UI
            if self.glados_engine_instance is not None:
                self.glados_worker = self.run_worker(self.glados_engine_instance.run, exclusive=True, thread=True)
                logger.info("GLaDOS worker started.")
            else:
                logger.error("Cannot start GLaDOS worker: glados_engine_instance is None.")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to start GLaDOS: {e}")

    def instantiate_glados(self) -> None:
        """
        Instantiate the GLaDOS engine.

        This function creates an instance of the GLaDOS engine, which is responsible for
        managing the GLaDOS system's operations and interactions. The instance can be used
        to control various aspects of the GLaDOS engine, including starting and stopping
        its event loop.

        Returns:
            Glados: An instance of the GLaDOS engine.
        """

        if not self.config_path.exists():
            logger.error(f"GLaDOS config file not found: {self.config_path}")
            return

        glados_config = GladosConfig.from_yaml(str(self.config_path))

        # Apply language override if provided
        if self.language:
            glados_config = glados_config.model_copy(update={"language": self.language.lower()})
            logger.info(f"Overriding ASR language to: {self.language}")

        self.glados_engine_instance = Glados.from_config(glados_config)

        def _post_init_bindings() -> None:
            self._attach_spectrum_stream()
            if self.pipeline_status_widget is not None:
                self.pipeline_status_widget.set_glados(self.glados_engine_instance)

        self.call_from_thread(_post_init_bindings)

    def _load_tui_settings(self) -> None:
        """Load additional TUI settings such as the login password."""
        self.login_password = None

        if not self.config_path.exists():
            logger.warning(f"TUI config file not found: {self.config_path}")
            return

        config_data: dict | None = None
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                config_data = yaml.safe_load(self.config_path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    logger.error(f"Could not decode TUI config file {self.config_path}")
                    return
            except OSError as exc:
                logger.error(f"Failed to read TUI config file {self.config_path}: {exc}")
                return
            except yaml.YAMLError as exc:
                logger.error(f"Failed to parse TUI config file {self.config_path}: {exc}")
                return

        if not isinstance(config_data, dict):
            logger.warning("Unexpected TUI config format; expected a mapping at top level.")
            return

        tui_config = config_data.get("tui")
        if isinstance(tui_config, dict):
            login_password = tui_config.get("login_password")
            if login_password is not None:
                self.login_password = str(login_password)
        else:
            logger.debug("No 'tui' section found in config; skipping TUI-specific settings.")

    def start_instantiation(self) -> None:
        """Starts the worker to instantiate the slow class."""
        if self.instantiation_worker is not None:
            self.notify("Instantiation already in progress!", severity="warning")
            return

        self.instantiation_worker = self.run_worker(
            self.instantiate_glados,  # The callable function
            thread=True,  # Run in a thread (default)
        )

    def _attach_spectrum_stream(self) -> None:
        if self.spectrum_widget and self.glados_engine_instance:
            self.spectrum_widget.set_data_queue(self.glados_engine_instance.spectrum_queue)

    @classmethod
    def run_app(cls, config_path: str | Path = "glados_config.yaml") -> None:
        app: GladosUI | None = None  # Initialize app to None
        try:
            app = cls()
            app.run()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user. Exiting.")
            if app is not None:
                app.exit()
            # No explicit sys.exit(0) here; Textual's app.exit() will handle it.
        except Exception:
            logger.opt(exception=True).critical("Unhandled exception in app run:")
            if app is not None:
                # Attempt a graceful shutdown even on other exceptions
                logger.info("Attempting graceful shutdown due to unhandled exception...")
                app.exit()
            sys.exit(1)  # Exit with error for unhandled exceptions


if __name__ == "__main__":
    GladosUI.run_app()
