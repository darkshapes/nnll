#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Orientations"""

from rich.highlighter import ReprHighlighter
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import ContentSwitcher, Footer, Static


from nnll_01 import debug_monitor
from nnll_10.package.display_bar import DisplayBar
from nnll_10.package.input_tag import InputTag
from nnll_10.package.message_panel import MessagePanel
from nnll_10.package.output_tag import OutputTag
from nnll_10.package.response_panel import ResponsePanel
from nnll_10.package.voice_panel import VoicePanel
from nnll_14 import build_conversion_graph, label_edge_attrib_for  # , trace_objective
from nnll_15.constants import LibType


class Fold(Screen):
    """Orienting display Horizontal
    Main interface container"""

    DEFAULT_CSS = """
    Screen {
        min-height: 5;
    }
    """
    TEXT = """"""

    BINDINGS = [
        Binding("`", "scribe_response", "go", priority=True),  # Send to LLM
        Binding("bk", "", "⌨️"),
        Binding("alt+bk", "clear_input", "del"),
        Binding("ent", "start_recording", "◉", priority=True),
        Binding("space", "play", "▶︎", priority=True),
        Binding("escape", "cancel_generation", "◼︎ / ⏏︎"),  # Cancel response
    ]

    nx_graph = None
    target_options: reactive[set] = reactive({})
    hilite = ReprHighlighter()
    input_map = {
        "text": "message_panel",
        "speech": "voice_panel",
        "image": "message_panel",
    }

    def compose(self) -> ComposeResult:
        """Create widgets"""
        # self.calculate_graph()
        self.calculate_graph()

        yield Footer(id="footer")
        with Horizontal(id="app-grid", classes="app-grid-horizontal"):
            yield ResponsiveLeftTop(id="left-frame")
            with Container(id="centre-frame"):
                with Container(id="responsive_input"):
                    with ContentSwitcher(id="panel_swap", initial="message_panel"):
                        yield MessagePanel(self.TEXT, id="message_panel", max_checkpoints=100)
                        yield VoicePanel(id="voice_panel")
                    yield InputTag(id="input_tag", classes="input_tag")
                yield DisplayBar(id="display_bar")
                with Container(id="responsive_display"):
                    yield ResponsePanel("\n", id="response_panel", language="markdown")
                    yield OutputTag(id="output_tag", classes="output_tag")
            yield ResponsiveRightBottom(id="right-frame")

    @work(exit_on_error=False)
    async def on_resize(self, event=events.Resize):
        """Fit shape to screen"""
        display = self.query_one("#app-grid")
        width = event.container_size.width
        height = event.container_size.height
        if width / 2 >= height:  # Screen is wide
            display.set_classes("app-grid-horizontal")
        elif width / 2 < height:  # Screen is tall
            display.set_classes("app-grid-vertical")

    # @work
    # async def on_ready(self) -> None:

    @work(exclusive=True)
    async def _on_key(self, event: events.Key):
        """Class method, window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            model_path = f"{self.target_options}"
            # target = self.query_one("#output_tag").target
            message = self.query_one("#message_panel").text
            # audio_sample = self.query_one("#voice_panel").audio
            # image_sample = self.query_one("#image_panel").file_name # probably drag and drop this
            # content = {
            #     "text": message if message and len(message) > 0 else None,
            #     "audio": audio_sample if audio_sample and len(audio_sample) > 0 else None,
            #     "image": image_sample if image_sample and len(image_sample) > 0 else None,
            # }
            self.query_one("#output_tag").add_class("active")
            response_panel = self.query_one("#response_panel")
            response_panel.scribe_response(model_path, message)  # , content, target)
            self.query_one("#output_tag").set_classes(["output_tag"])
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
        # self.current_model = self.query_one("#output_tag").current_model
        # self.query_one("#display_bar").calculate_tokens(self.current_model, message)

    @work(exclusive=True)
    async def pass_audio_to_tokenizer(self) -> None:
        """Transmit audio to sample length"""
        sample_length = len(self.query_one("#voice_panel").audio)
        sample_frequency = self.query_one("#voice_panel").sample_freq
        duration = float(sample_length / sample_frequency) if sample_length > 1 else 0.0
        self.query_one("#display_bar").calculate_audio(duration)

    @debug_monitor
    def _on_mouse_scroll_down(self, event: events.MouseScrollUp) -> None:
        """Determine tag_name focus by negative space, then trigger scroll down at 1/10th intensity"""
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            output_tag = self.query_one("#output_tag")
            self.target_options = output_tag.emulate_scroll_down(output_tag.target_options)
        elif self.query_one("#input_tag").has_focus:
            event.prevent_default()
            input_tag = self.query_one("#input_tag")
            class_name = input_tag.emulate_scroll_down(input_tag.target_options)
            self.query_one(ContentSwitcher).current = self.input_map.get(class_name)

    @debug_monitor
    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Determine tag_name focus by negative space, then trigger scroll down at 1/10th intensity"""
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            output_tag = self.query_one("#output_tag")
            self.target_options = output_tag.emulate_scroll_up(output_tag.target_options)
        elif self.query_one("#input_tag").has_focus:
            event.prevent_default()
            input_tag = self.query_one("#input_tag")
            class_name = input_tag.emulate_scroll_up(input_tag.target_options)
            self.query_one(ContentSwitcher).current = self.input_map.get(class_name)

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        """Stop the processing of a model"""
        self.query_one("#response_panel").workers.cancel_all()
        self.query_one("#output_tag").set_classes("output_tag")

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
        self.query_one(ContentSwitcher).current = id_name
        # self.query_one("#voice_panel").focus()

    def calculate_graph(self):
        self.nx_graph = build_conversion_graph()
        self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.HUB)
        self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.OLLAMA)
        return self.nx_graph


class ResponsiveLeftTop(Container):
    """Sidebar Left/Top"""

    def compose(self) -> ComposeResult:
        yield Static()


class ResponsiveRightBottom(Container):
    """Sidebar Right/Bottom"""

    def compose(self) -> ComposeResult:
        yield Static()
        yield Static()
