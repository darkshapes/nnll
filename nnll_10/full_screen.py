#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from textual import events, on, work
from textual.binding import Binding
from textual.app import ComposeResult
from textual.widgets import Footer, TextArea
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from textual.screen import Screen

from .entry_field import EntryField
from .read_out import ReadOut  # pylint: disable=import-error
from .display_bar import DisplayBar  # pylint: disable=import-error
from .tag_line import TagLine  # pylint: disable=import-error
from .text_generator import chat_machine


class CentreFold(Vertical):
    """Foldout interface"""  # machine output

    CSS_PATH = "combo.tcss"


class ResponsiveDisplay(Horizontal):
    """Read-only interface components"""

    CSS_PATH = "combo.tcss"


class FullScreen(Screen):
    """Central machine interface"""

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
        Binding("escape", "exit_now()", "Leave Focus", priority=True),  # Quit out
    ]
    CSS_PATH = "combo.tcss"

    safety: reactive[int] = reactive(5)

    def compose(self) -> ComposeResult:
        yield Footer()
        with Container(id="DisplayArea", name="DisplayArea", classes="DisplayArea"):
            yield Container(id="SidebarLeft", name="SidebarLeft", classes="SidebarLeft")
            with CentreFold(name="CentreFold"):
                yield EntryField(self.TEXT, max_checkpoints=100, theme="vscode_dark", language="python")
                yield DisplayBar(name="DisplayBar")
                # # Sparkline(self.tokens, summary_function=max)
                with ResponsiveDisplay(name="ResponsiveDisplay"):
                    yield ReadOut("", name="ReadOut", language="markdown", read_only=True)
                    yield TagLine(text=self.status, name="TagLine", url="", id="TagLine")  # Show system state floating object
            yield Container(id="SidebarRight", name="SidebarRight", classes="SidebarRight")

    def on_mount(self):
        display_bar = self.query_one(DisplayBar)
        display_bar.add_columns(*self.rows[0])
        display_bar.add_rows(self.rows[1:])
        read_out = self.query_one(ReadOut)
        read_out.soft_wrap = True
        read_out.line_nubmber_start = 3
        tag_line = self.query_one(TagLine)
        tag_line.styles.text_style = "dim"
        tag_line.status = f"{self.model_short}"

    @work(exit_on_error=False)
    async def _on_key(self, event: events.Key):
        """Window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            # await self.action_post_message()
        elif event.key != "escape":
            self.safety = 3
        else:
            event.prevent_default()
            self.safety -= 1
            if self.safety <= 0:
                event.prevent_default()
                self.exit_now()

    @work(exit_on_error=True)
    async def exit_now(self) -> None:
        """Immediately exit the app"""
        self.app.exit()
        self.app.action_quit()

    @work(exit_on_error=False)
    async def get_token_count(self, message):
        """Live token count"""
        display_bar = self.query_one(DisplayBar)
        display_bar.calculate_tokens(message)

    # @work(exit_on_error=False)
    # async def action_post_message(self) -> None:
    #     """Pull entry text and generate from it"""
    #     entry_field = self.query_one(EntryField)
    #     message = await entry_field.read_text_field()
    #     if message.lower() == "quit" or message.lower() == "exit":
    #         self.app.exit()

    #     read_out = self.query_one(ReadOut)
    #     read_out.generate_response(message, self.model)

    # @work(exit_on_error=False, name="chat", group="chat_worker")
    # async def generate_response(self, message, model):
    #     """Fill display with generated content"""
    #     read_out = self.query_one(ReadOut)
    #     read_out = await read_out.prepare_display_log()
    #     async for chunk in chat_machine(model, message):
    #         read_out.insert(chunk)

    # @work(exit_on_error=False)
    # async def cancel_generation(self) -> None:
    #     await self.workers.cancel_group(node="TextStream", group="chat_worker")
    #     ticker = self.query_one(TagLine)
    #     await ticker.update_status(TagLine)


# class CentreFold(Vertical):
#     pass
