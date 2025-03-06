#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Main Processing Module"""

import os
from typing import Callable

from textual import events, work, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import TextArea, DataTable
from textual.containers import Container
from textual.reactive import reactive
import tiktoken
from fetch_models import from_ollama_cache  # pylint: disable=import-error
from chat_machine import chat_machine  # pylint: disable=import-error


class MessagePanel(TextArea):
    pass


class Fold(Container):
    """Main interface container"""

    TAG_ACTIVE_HUE: reactive[str] = reactive("orangered")
    TAG_PASSIVE_HUE: reactive[str] = reactive("magenta")
    TEXT = """"""
    BINDINGS = [
        Binding("`", "generate_response", "Send", priority=True),  # Send to LLM
        Binding("escape", "cancel_generation", "Cancel", priority=True),  # Cancel response
    ]
    UNIT1 = "chr /"  # Display Bar Units
    UNIT2 = "tkn /"
    rows: list[tuple] = [
        (0, 0),
        (f"     0{UNIT1}   ", f"0{UNIT2}   "),
    ]
    line_start: reactive[str] = reactive(3, init=True)
    available_models: reactive[dict] = reactive({})
    current_model: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield MessagePanel(self.TEXT, id="message_panel", max_checkpoints=100, theme="vscode_dark", language="python")
        yield DataTable(id="display_bar", show_header=False, show_row_labels=False, cursor_type=None)
        with Container(name="responsive_display"):
            yield TextArea("", id="response_panel", language="markdown", read_only=True)
            yield DataTable(id="tag_line", show_header=False, show_row_labels=False, show_cursor=False)  # Show system state floating object

    def on_mount(self) -> None:
        """Class method, initialize"""
        message_panel = self.query_one("#message_panel", MessagePanel)
        message_panel.border_subtitle = "Prompt"
        message_panel.styles.border_subtitle_color = "mediumturquoise"
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.add_columns(*self.rows[0])
        display_bar.add_rows(self.rows[1:])
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.soft_wrap = True
        response_panel.line_number_start = self.line_start
        tag_line = self.query_one("#tag_line", DataTable)
        self.available_models = from_ollama_cache()
        tag_line.add_columns(("0", "1"))
        tag_line.add_rows([row.strip()] for row in self.available_models)
        tag_line.styles.text_overflow = "ellipsis"
        tag_line.cursor_foreground_priority = "renderable"
        tag_line.cell_padding = 0

        tag_line.styles.overflow_x = "hidden"

    @work(exclusive=True)
    async def _on_key(self, event: events.Key) -> Callable:
        """Class method, window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            self.generate_response()

    @work(group="chat")
    async def generate_response(self):
        """Fill display with generated content"""
        message_panel = self.query_one(MessagePanel)
        message = message_panel.text
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.insert("---\n")
        response_panel.move_cursor(response_panel.document.end)
        response_panel.scroll_end(animate=True)
        self.update_status()
        async for chunk in chat_machine(self.current_model, message):
            response_panel.insert(chunk)
        self.update_status()

    @on(TextArea.Changed)
    @work(exclusive=True, group="token")
    async def calculate_tokens(self) -> None:
        """Called when message input area is manipulated.
        Live display of tokens and characters"""
        message_panel = self.query_one("#message_panel", TextArea)
        message = message_panel.text
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(message))
        character_count = len(message)
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.update_cell_at((0, 0), f"     {character_count}{self.UNIT1}")
        display_bar.update_cell_at((0, 1), f"{token_count}{self.UNIT2}")

    @work(exclusive=True)
    async def update_status(self) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", DataTable)
        style = tag_line.get_visual_style(partial=True)
        tag_line.styles.color = "magenta" if style.foreground == "darkorange" else "darkorange"

    @work(exclusive=True)
    async def update_model(self) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", DataTable)
        current_model = tag_line.coordinate_to_cell_key(tag_line.cursor_coordinate)
        self.current_model = current_model

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        # response_panel = self.query_one("#response_panel", TextArea)
        await self.workers.cancel_group(node="#response_panel", group="chat")
        self.update_status()

    # @work(exclusive=True)
    # async def on_paste(self) -> None:  # ,  event: events.Paste):
    #     """Class method, attempting to stop token generation on paste"""
    #     display_bar = self.query_one("#display_bar", DataTable)
    #     await self.workers.cancel_group(node=display_bar, group="token")

    # Sparkline ticker to recycle
    # Sparkline(self.tokens, summary_function=max
    #  count = var(0) part of sparkline
    # offset = self.count * 120
    # self.tokens.e
    # offset = self.count
    # for x in self.tokens:
    #     data_test = range(0 + offset, len(self.tokens), x)
    # stream = [x for x in self.tokens[]]
    # self.query_one(Sparkline).data = [self.tokens]
    # response_panel.border_title = self.status
    # response_panel.styles.border_title_color = "magenta"
