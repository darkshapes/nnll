#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Central Processing Module"""

import os
from typing import Callable

from textual import events, work, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.widgets import TextArea, DataTable, Link
from textual.containers import Container
from textual.reactive import reactive
import tiktoken

from chat_machine import chat_machine


class MessagePanel(TextArea):
    pass


class CentreFold(Container):
    #     """Central interface"""

    CSS_PATH = "centre_fold.tcss"

    model = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"  # example default (drag/drop model to load from server?)
    model_short = os.path.basename(model)  # shorter for display
    status: reactive[str] = reactive(f"{model_short}")  # mediumvioletred # Chromatic status indication
    UNIT1 = "chr /"  # Display Bar Units
    UNIT2 = "tkn /"
    rows: list[tuple] = [
        (0, 0),
        (f"      {UNIT1}   ", f" {UNIT2}   "),
    ]
    TEXT = """Prompt"""
    BINDINGS = [
        Binding("`", "generate_response", "Send", priority=True),  # Send to LLM
        Binding("escape", "cancel_generation", "Cancel", priority=True),  # Cancel response
    ]

    def compose(self) -> ComposeResult:
        yield MessagePanel(self.TEXT, id="message_panel", max_checkpoints=100, theme="vscode_dark", language="python")
        yield DataTable(id="display_bar")

        with Container(name="responsive_display"):
            yield TextArea("", id="response_panel", language="markdown", read_only=True)
            yield Link(self.status, id="tag_line", url="None")  # Show system state floating object

    def on_mount(self) -> None:
        """Class method, initialize"""
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.add_columns(*self.rows[0])
        display_bar.add_rows(self.rows[1:])
        display_bar.styles.background = Color(0, 0, 0)
        display_bar.styles.back = Color(0, 0, 0)
        display_bar.styles.color = "#333333"
        display_bar.show_row_labels = False
        display_bar.show_header = False
        display_bar.cursor_type = None
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.soft_wrap = True
        response_panel.line_number_start = 3
        tag_line = self.query_one("#tag_line", Link)
        tag_line.styles.text_style = "dim"
        tag_line.status = f"{self.model_short}"

    @work(exclusive=True)
    async def _on_key(self, event: events.Key) -> Callable:
        """Class method, window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            self.generate_response(self.model)

    @work(exclusive=True)
    async def read_text_field(self):
        """Return all the text in the widget"""
        message_panel = self.query_one(MessagePanel)
        if message_panel.text.lower() == "quit" or message_panel.text.lower() == "exit":
            self.app.exit()
        message = message_panel.text
        return message

    @work(group="chat")
    async def generate_response(self, model):
        """Fill display with generated content"""
        message = self.read_text_field
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel = self.prepare_display_log()
        self.update_status()
        async for chunk in chat_machine(model, message):
            response_panel.insert(chunk)
        self.update_status()

    @on(TextArea.Changed)
    @work(exclusive=True, group="token")
    async def calculate_tokens(self) -> None:
        """Called when message input area is manipulated.
        Live display of tokens and characters"""

        message_panel = self.query_one("#message_panel", TextArea)
        encoding = tiktoken.get_encoding("cl100k_base")
        message = message_panel.text
        token_count = len(encoding.encode(message))
        character_count = len(message)
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.update_cell_at((0, 0), f"     {character_count}{self.UNIT1}")
        display_bar.update_cell_at((0, 1), f"{token_count}{self.UNIT2}")

    @work(exclusive=True)
    async def prepare_display_log(self):
        """Ready display for new text"""
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.insert("---\n")
        response_panel.move_cursor(response_panel.document.end)
        response_panel.scroll_end(animate=True)

    @work(exclusive=True)
    async def update_status(self) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", Link)
        tag_line.styles.color = "magenta" if tag_line.styles.color == "green" else "green"

    @work(exclusive=True)
    async def update_model(self, model: str) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", Link)
        tag_line.model_short = model

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        # response_panel = self.query_one("#response_panel", TextArea)
        await self.workers.cancel_group(node="#response_panel", group="chat")
        tag_line = self.query_one("#tag_line", Link)
        await tag_line.update_status()

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
