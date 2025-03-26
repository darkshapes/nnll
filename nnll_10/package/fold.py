#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Main Processing Module"""

from rich.highlighter import ReprHighlighter
from textual import events, work, on
from textual.app import ComposeResult

from textual.binding import Binding
from textual.reactive import reactive
from textual.containers import Container
from textual.widgets import ContentSwitcher


# from textual.reactive import reactive
from nnll_01 import debug_monitor
from nnll_10.package.display_bar import DisplayBar

from nnll_10.package.message_panel import MessagePanel
from nnll_10.package.response_panel import ResponsePanel
from nnll_10.package.tag_line import TagLine
from nnll_10.package.input_tag import InputTag
from nnll_10.package.voice_panel import VoicePanel

# from package.panel_swap import PanelSwap


class PanelSwap(ContentSwitcher):
    """Switch media panel"""


class Fold(Container):
    """Main interface container"""

    TEXT = """"""

    BINDINGS = [
        Binding("`", "generate_response", "go", priority=True),  # Send to LLM
        Binding("bk", "", "⌨️"),
        Binding("alt+bk", "clear_input", "del"),
        Binding("ent", "start_recording", "◉", priority=True),
        Binding("space", "play", "▶︎", priority=True),
        Binding("escape", "cancel_generation", "◼︎ / ⏏︎"),  # Cancel response
    ]

    UNIT1 = "chr /   "  # Display Bar Units
    UNIT2 = "tkn / "
    UNIT3 = "″"
    unit_labels = [UNIT1, UNIT2, UNIT3]
    rows: list[tuple] = [
        (0, 0, 0, 0),
        (f"     0{UNIT1}", f"0{UNIT2}", f"0.0{UNIT3}", " "),
    ]

    current_model: reactive[str] = reactive("")
    hilite = ReprHighlighter()

    def compose(self) -> ComposeResult:
        with Container(id="responsive_input"):
            with PanelSwap(id="panel_swap", initial="message_panel"):
                yield MessagePanel(self.TEXT, id="message_panel", max_checkpoints=100)
                yield VoicePanel(id="voice_panel")
            yield InputTag(id="input_tag")
        yield DisplayBar(id="display_bar")
        with Container(id="responsive_display"):
            yield ResponsePanel("\n", id="response_panel")
            yield TagLine(id="tag_line", classes="tag_line")

    def on_mount(self) -> None:
        """Class method, initialize"""
        self.current_model = self.query_one("#tag_line").current_model
        display_bar = self.query_one("#display_bar")
        display_bar.add_columns(*self.rows[0])
        display_bar.add_rows(self.rows[1:])

    @work(exclusive=True)
    async def _on_key(self, event: events.Key):
        """Class method, window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            message = self.query_one("#message_panel").text
            model_path = f"{self.current_model}"
            response_panel = self.query_one("#response_panel")
            self.query_one("#tag_line").add_class("active")
            response_panel.generate_response(model_path, message)
            self.query_one("#tag_line").set_classes("tag_line")
        elif event.key == "escape":  # self.query_one("#responsive_input").has_focus_within and
            # self.query_one("#response_panel").focus()
            self.cancel_generation()
        elif (hasattr(event, "character") and event.character == "\r") or event.key == "enter":
            self.alternate_panel("voice_panel", 1)
            voice_panel = self.query_one("#voice_panel")
            voice_panel.record_audio()
            self.pass_audio_to_tokenizer()
        elif (hasattr(event, "character") and event.character == " ") or event.key == "space":
            self.alternate_panel("voice_panel", 1)
            self.query_one("#voice_panel").play_audio()
        elif (event.name) == "ctrl_w" or event.key == "ctrl+w":
            self.clear_input()
        elif (hasattr(event, "character") and event.character == "\x7f") or event.key == "backspace":
            if not self.query_one("#response_panel").has_focus:
                self.alternate_panel("message_panel", 0)

    @work(exclusive=True)
    @on(MessagePanel.Changed, "#message_panel")
    async def pass_text_to_tokenizer(self) -> None:
        """Transmit info to token calculation"""
        message = self.query_one("#message_panel").text
        self.current_model = self.query_one("#tag_line").current_model
        self.query_one("#display_bar").calculate_tokens(self.current_model, message, self.unit_labels)

    @work(exclusive=True)
    async def pass_audio_to_tokenizer(self) -> None:
        """Transmit audio to sample length"""
        sample_length = len(self.query_one("#voice_panel").audio)
        sample_frequency = self.query_one("#voice_panel").sample_freq
        duration = float(sample_length / sample_frequency)
        self.query_one("#display_bar").calculate_audio(duration, self.unit_labels)

    @debug_monitor
    def _on_mouse_scroll_down(self, event: events.MouseScrollUp) -> None:
        """Determine tag_name focus by negative space, then trigger scroll down at 1/10th intensity"""
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            tag_line = self.query_one("#tag_line")
            current_model = tag_line.emulate_scroll_down(tag_line.available_models)
            self.current_model = current_model
        elif self.query_one("#input_tag").has_focus:
            event.prevent_default()
            input_tag = self.query_one("#input_tag")
            class_name = input_tag.emulate_scroll_down(input_tag.available_inputs)
            self.query_one(PanelSwap).current = class_name

    @debug_monitor
    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Determine tag_name focus by negative space, then trigger scroll down at 1/10th intensity"""
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            tag_line = self.query_one("#tag_line")
            current_model = tag_line.emulate_scroll_up(tag_line.available_models)
            self.current_model = current_model
        elif self.query_one("#input_tag").has_focus:
            event.prevent_default()
            input_tag = self.query_one("#input_tag")
            class_name = input_tag.emulate_scroll_up(input_tag.available_inputs)
            self.query_one(PanelSwap).current = class_name

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        """Stop the processing of a model"""
        self.query_one("#response_panel").workers.cancel_all()
        self.query_one("#tag_line").set_classes("tag_line")

    @work(exclusive=True)
    async def clear_input(self):
        """Clear the input on the focused panel"""
        if self.query_one("#voice_panel").has_focus:
            self.query_one("#voice_panel").erase_audio()
            self.pass_audio_to_tokenizer()
        elif self.query_one("#message_panel").has_focus:
            self.query_one("#message_panel").erase_message()

    @work(exclusive=True)
    async def alternate_panel(self, id_name, y_coordinate):
        input_tag = self.query_one("#input_tag")
        input_tag.scroll_to(x=0, y=y_coordinate, force=True, immediate=True, on_complete=input_tag.refresh)
        self.query_one(PanelSwap).current = id_name
        # self.query_one("#voice_panel").focus()
