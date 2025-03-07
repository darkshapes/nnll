#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Main Processing Module"""

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
    model_cell: reactive[int] = reactive(0.0)
    available_models: reactive[dict] = reactive({})
    current_model: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield MessagePanel(self.TEXT, id="message_panel", max_checkpoints=100, theme="vscode_dark", language="python")
        yield DataTable(id="display_bar", show_header=False, show_row_labels=False, cursor_type=None)
        with Container(id="responsive_display"):
            yield TextArea("", id="response_panel", language="markdown", read_only=True)
            # yield SpinBox([model for model in from_ollama_cache()], id="tag_line")  # Show system state floating object
            yield DataTable(id="tag_line", show_header=False)

    def on_mount(self) -> None:
        """Class method, initialize"""
        message_panel = self.query_one("#message_panel")
        message_panel.border_subtitle = "Prompt"
        message_panel.styles.border_subtitle_color = "mediumturquoise"
        display_bar = self.query_one("#display_bar")
        display_bar.add_columns(*self.rows[0])
        display_bar.add_rows(self.rows[1:])
        response_panel = self.query_one("#response_panel")
        response_panel.soft_wrap = True
        response_panel.insert("\n")
        tag_line = self.query_one("#tag_line")
        self.available_models = from_ollama_cache()
        tag_line.add_columns(("0", "1"))
        tag_line.add_rows([row.strip()] for row in self.available_models)
        tag_line.styles.text_overflow = "ellipsis"
        tag_line.cursor_type = "cell"

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
        response_panel = self.query_one("#response_panel")
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
        message_panel = self.query_one("#message_panel")
        message = message_panel.text
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(message))
        character_count = len(message)
        display_bar = self.query_one("#display_bar")
        display_bar.update_cell_at((0, 0), f"     {character_count}{self.UNIT1}")
        display_bar.update_cell_at((0, 1), f"{token_count}{self.UNIT2}")

    @work(exclusive=True)
    async def update_status(self) -> None:
        """Provide visual status update of processing"""
        tag_line = self.query_one("#tag_line")
        style = tag_line.get_visual_style(partial=True)
        tag_line.styles.color = "magenta" if style.foreground == "darkorange" else "darkorange"

    def _on_mouse_scroll_down(self, event: events.MouseScrollUp) -> None:
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            if self.model_cell < len(list(self.available_models)) - 1:
                self.model_cell += 0.1
                self.tag_line_scroller()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.query_one("#responsive_display").has_focus_within != self.query_one("#response_panel").has_focus:
            event.prevent_default()
            if self.model_cell > 0:
                self.model_cell -= 0.1
                self.tag_line_scroller()

    @work(exclusive=True)
    async def tag_line_scroller(self):
        coordinate = int(round(self.model_cell))
        tag_line = self.query_one("#tag_line")
        tag_line.move_cursor(row=coordinate, column=0)
        key_name = tag_line.get_cell_at((coordinate, 0))
        self.current_model = self.available_models.get(key_name)

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        await self.workers.cancel_group(node="#response_panel", group="chat")
        self.update_status()


# @on(DataTable.CellHighlighted)
# async def update_selection(self, event: events) -> None:
#     tag_line = self.query_one("#tag_line")
#     response_panel = self.query_one("#response_panel")
#     response_panel.insert(str(tag_line.CellSelected))
# tag_line.styles.overflow_x = "hidden"
# tag_line.styles.overflow_x = "hidden"
# for model in self.available_models.keys():
# tag_line.write(model, shrink=True)

# response_panel.insert(str(tag_line.cursor_row))

#     response_panel.insert(str(tag_line.value))
#     # tag_line.validate_scroll_y
#     # current_model = tag_line.coordinate_to_cell_key(tag_line.scroll_y)
#     # self.current_model = current_model

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

# response_panel = self.query_one("#response_panel")
# response_panel.insert(str(coordinate))
# data = self.query_one("#tag_line").columns[key_name]
# response_panel.insert(self.current_model)
# @on(events.Enter, "#message_panel")
# @work(exclusive=True)
# async def focus_message_panel(self, event: events) -> None:
#     """Provide visual status update of processing"""
#     message_panel = self.query_one("#message_panel")
#     if "Enter" in str(event):
#         message_panel.focus(True)
#     else:
#         message_panel.focus(False)

# @on(events.Enter, "#response_panel")
# @work(exclusive=True)
# async def focus_response_panel(self, event: events) -> None:
#     """Provide visual status update of processing"""
#     response_panel = self.query_one("#response_panel")
#     if "Enter" in str(event):
#         response_panel.focus(True)
#     else:
#         response_panel.focus(False)

# @on(events.Enter, "#tag_line")
# @on(events.Leave, "#tag_line")
# async def focus_tag_line(self, event: events) -> None:
#     """Provide visual status update of processing"""
#     tag_line = self.query_one("#tag_line")
#     response_panel = self.query_one("#response_panel")
#     if "Enter" in str(event):
#         tag_line.focus(True)
#         response_panel.insert(str("enter"))
#     elif "Leave" in str(event):
#         tag_line.focus(False)
#         response_panel.insert(str("leave"))
