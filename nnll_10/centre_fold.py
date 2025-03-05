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
from .chat_machine import chat_machine


class CentreFold(Container):
    #     """Central interface"""

    CSS_PATH = "combo.tcss"

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
        Binding("`", "", "Send", priority=True),  # Send to LLM
    ]
    # CSS_PATH = "combo.tcss"

    def compose(self) -> ComposeResult:
        yield TextArea(self.TEXT, id="message_panel", max_checkpoints=100, theme="vscode_dark", language="python")
        yield DataTable(id="display_bar")

        # Sparkline(self.tokens, summary_function=max
        with Container(name="responsive_display"):
            yield TextArea("", id="response_panel", language="markdown", read_only=True)
            yield Link(self.status, id="tagline", url="None")  # Show system state floating object

    def on_mount(self) -> None:
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
        tag_line = self.query_one("#tagline", Link)
        tag_line.styles.text_style = "dim"
        tag_line.status = f"{self.model_short}"

    @work(exit_on_error=False)
    async def _on_key(self, event: events.Key) -> Callable:
        """Window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            await self.action_post_message()

    @work(exit_on_error=False)
    async def read_text_field(self):
        """Return all the text in the widget"""
        message_panel = self.query_one("#message_panel", TextArea)
        message = message_panel.text
        return message

    @work(exit_on_error=False)
    async def _on_key(self, event: events.Key):
        """Window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            self.read_text_field()

    @on(TextArea.Changed)
    async def update_tokens(self):
        return self.read_text_field()

    @work(exit_on_error=False)
    async def get_token_count(self, message):
        """Live token count"""
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.calculate_tokens(message)

    @work(exit_on_error=False)
    async def action_post_message(self) -> None:
        """Pull entry text and generate from it"""
        entry_field = self.query_one("#message_panel", TextArea)
        message = await entry_field.read_text_field()
        if message.lower() == "quit" or message.lower() == "exit":
            self.app.exit()

        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.generate_response(message, self.model)

    @work(exit_on_error=False, name="chat", group="chat_worker")
    async def generate_response(self, message, model):
        """Fill display with generated content"""
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel = await response_panel.prepare_display_log()
        async for chunk in chat_machine(model, message):
            response_panel.insert(chunk)

    @work(exit_on_error=False)
    async def cancel_generation(self) -> None:
        response_panel = self.query_one("#response_panel", TextArea)
        await self.workers.cancel_node(node=response_panel)
        tag_line = self.query_one("#tag_line", Link)
        await tag_line.update_status()

    async def calculate_tokens(self, message):  # field: TextArea.Changed):
        """Live display of tokens and characters"""
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(message))
        character_count = len(message)
        display_bar = self.query_one("#display_bar", DataTable)
        display_bar.update_cell_at((0, 0), f"     {character_count}{self.UNIT1}")
        display_bar.update_cell_at((0, 1), f"{token_count}{self.UNIT2}")

    @work(exit_on_error=False)
    async def prepare_display_log(self):
        """Ready display for new text"""
        response_panel = self.query_one("#response_panel", TextArea)
        response_panel.insert("---\n")
        response_panel.move_cursor(response_panel.document.end)
        response_panel.scroll_end(animate=True)

    @work(exit_on_error=False)
    async def update_status(self) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", Link)
        tag_line.styles.color = "magenta" if tag_line.styles.color == "bold green" else "bold green"

    @work(exit_on_error=False)
    async def update_model(self, model: str) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line", Link)
        tag_line.model_short = model
