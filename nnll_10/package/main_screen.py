#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Auto-Orienting Split screen"""

from collections import defaultdict
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container

# from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static, ContentSwitcher

from nnll_01 import debug_monitor
from nnll_05 import pull_path_entries
from nnll_10.package.message_panel import MessagePanel
from nnll_14 import calculate_graph


class Fold(Screen[bool]):
    """Orienting display Horizontal
    Main interface container"""

    DEFAULT_CSS = """Screen { min-height: 5; }"""

    BINDINGS = [
        Binding("bk", "", "⌨️"),
        Binding("alt+bk", "clear_input", "del"),
        Binding("ent", "start_recording", "◉", priority=True),
        Binding("space", "play", "▶︎", priority=True),
        Binding("escape", "cancel_generation", "◼︎ / ⏏︎"),  # Cancel response
        Binding("`", "scribe_response", "✎", priority=True),  # Send to LLM
    ]

    foldr = defaultdict(dict)
    nx_graph = calculate_graph()
    traced_path: list = []
    regitry_entries: list = []
    next_model: str = ""
    input_map = {
        "text": "message_panel",
        "image": "message_panel",
        "speech": "voice_panel",
    }

    def compose(self) -> ComposeResult:
        """Construct custom widget classes"""
        from textual.containers import Horizontal
        from textual.widgets import Footer
        from nnll_10.package.display_bar import DisplayBar
        from nnll_10.package.input_tag import InputTag
        from nnll_10.package.output_tag import OutputTag
        from nnll_10.package.response_panel import ResponsePanel
        from nnll_10.package.voice_panel import VoicePanel

        yield Footer(id="footer")
        with Horizontal(id="app-grid", classes="app-grid-horizontal"):
            yield ResponsiveLeftTop(id="left-frame")
            with Container(id="centre-frame"):
                with Container(id="responsive_input"):
                    with ContentSwitcher(id="panel_swap", initial="message_panel"):
                        yield MessagePanel("""""", id="message_panel", max_checkpoints=100)
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

    @work(exclusive=True)
    async def on_mount(self):
        """Class function, query all at once"""
        self.foldr["db"] = self.query_one("#display_bar")
        self.foldr["it"] = self.query_one("#input_tag")
        self.foldr["mp"] = self.query_one("#message_panel")
        self.foldr["ot"] = self.query_one("#output_tag")  # type : ignore
        self.foldr["ps"] = self.query_one(ContentSwitcher)
        self.foldr["rd"] = self.query_one("#responsive_display")
        self.foldr["rp"] = self.query_one("#response_panel")
        self.foldr["vp"] = self.query_one("#voice_panel")
        # id_name = self.input_tag.highlight_link_id

    @work(exclusive=True)
    async def on_focus(self, event: events.Focus):
        from nnll_14 import trace_objective

        in_type = self.foldr["it"].get_cell_at((int(round(self.foldr["it"].content_cell)), 1))
        out_type = self.foldr["ot"].get_cell_at((int(round(self.foldr["it"].content_cell)), 1))
        self.traced_path = trace_objective(nx_graph=self.nx_graph, source=in_type, target=out_type)
        self.registry_entries = pull_path_entries(self.nx_graph, self.traced_path)
        self.next_model = next(iter([x["entry"].model for x in list(self.registry_entries)]))

    @work(exclusive=True)
    async def _on_key(self, event: events.Key):
        """Class method event trigger
        Suppress/augment default key actions to trigger keybindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            self.machine_handoff()
        elif event.key == "escape" and "active" in self.foldr["ot"].classes:
            self.cancel_generation()
        elif (hasattr(event, "character") and event.character == "\r") or event.key == "enter":
            self.alternate_panel("voice_panel", 1)
            self.foldr["vp"].record_audio()
            self.pass_audio_to_tokenizer()
        elif (hasattr(event, "character") and event.character == " ") or event.key == "space":
            self.alternate_panel("voice_panel", 1)
            self.foldr["vp"].play_audio()
        elif (event.name) == "ctrl_w" or event.key == "ctrl+w":
            self.clear_input()
        elif not self.foldr["rp"].has_focus and ((hasattr(event, "character") and event.character == "\x7f") or event.key == "backspace"):
            self.alternate_panel("message_panel", 0)

    @debug_monitor
    def _on_mouse_scroll_down(self, event: events.MouseScrollUp) -> None:
        """Class method event trigger
        Translate scroll events into datatable cursor movement
        Trigger scroll at 1/10th intensity when menu has focus
        :param event: Event data for the trigger"""

        if self.foldr["rd"].has_focus_within != self.foldr["rp"].has_focus:
            event.prevent_default()
            self.foldr["ot"].emulate_scroll_down()
        elif self.foldr["it"].has_focus:
            event.prevent_default()
            class_name = self.foldr["it"].emulate_scroll_down()
            self.foldr["ps"].current = self.input_map.get(class_name)

    @debug_monitor
    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Class method event trigger
        Translate scroll events into datatable cursor movement
        Trigger scroll at 1/10th intensity when menu has focus
        :param event: Event data for the trigger"""

        if self.foldr["rd"].has_focus_within != self.foldr["rp"].has_focus:
            event.prevent_default()
            self.foldr["ot"].emulate_scroll_up()
        elif self.foldr["it"].has_focus:
            event.prevent_default()
            class_name = self.foldr["it"].emulate_scroll_up()
            self.foldr["ps"].current = self.input_map.get(class_name)

    @work(exclusive=True)
    @on(MessagePanel.Changed, "#message_panel")
    async def pass_text_to_tokenizer(self) -> None:
        """Transmit info to token calculation"""
        message = self.foldr["mp"].text
        self.foldr["db"].calculate_tokens(self.next_model, message)

    @work(exclusive=True)
    async def pass_audio_to_tokenizer(self) -> None:
        """Transmit audio to sample length"""
        duration = self.foldr["vp"].calculate_sample_length()
        self.foldr["db"].calculate_audio(duration)

    @work(exclusive=True)
    async def machine_handoff(self) -> None:
        """Provide prompts to generative processing endpoint"""
        content = {
            "text": self.foldr["mp"].text,
            "audio": self.foldr["vp"].audio,
            # "attachment": self.message_panel.file # drag and drop from external window
            # "image": self.image_panel.image #  active image feed from cam / import screen
        }
        self.foldr["ot"].add_class("active")
        self.foldr["rp"].scribe_response(self.nx_graph, content, target=self.foldr["ot"].target_options)
        self.foldr["ot"].set_classes(["output_tag"])

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        """Stop the processing of a model"""
        self.foldr["rp"].workers.cancel_all()
        self.foldr["ot"].set_classes("output_tag")

    @work(exclusive=True)
    async def clear_input(self) -> None:
        """Clear the input on the focused panel"""
        if self.foldr["vp"].has_focus:
            self.foldr["vp"].erase_audio()
            self.pass_audio_to_tokenizer()
        elif self.foldr["mp"].has_focus:
            self.foldr["mp"].erase_message()

    @work(exclusive=True)
    async def alternate_panel(self, id_name: str, y_coordinate: int) -> None:
        """Switch between text input and audio input
        :param id_name: The panel to switch to
        :param y_coordinate: _description_
        """
        self.foldr["it"].scroll_to(x=1, y=y_coordinate, force=True, immediate=True, on_complete=self.foldr["it"].refresh)
        self.foldr["ps"].current = id_name


class ResponsiveLeftTop(Container):
    """Sidebar Left/Top"""

    def compose(self) -> ComposeResult:
        yield Static()


class ResponsiveRightBottom(Container):
    """Sidebar Right/Bottom"""

    def compose(self) -> ComposeResult:
        yield Static()
        yield Static()
