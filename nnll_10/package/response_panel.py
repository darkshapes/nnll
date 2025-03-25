#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work

# from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import TextArea
from nnll_01 import debug_monitor
from nnll_10.package.chat_machine import chat_machine


class ResponsePanel(TextArea):
    """Machine response field"""

    prefix: str = "ollama_chat/"
    is_generating: reactive[bool] = reactive(False)

    def on_mount(self):
        self.language = "markdown"
        self.read_only = True
        self.soft_wrap = True

    @debug_monitor
    @work(group="chat")
    async def generate_response(self, model, message):
        """Fill display with generated content"""
        self.is_generating = True
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")
        model = f"{self.prefix}{model}"
        self.move_cursor(self.document.end, center=True)
        try:
            async for chunk in chat_machine(model, message):
                self.insert(chunk)
                self.scroll_cursor_visible(center=True, animate=True)
        except AttributeError as error_log:
            print(error_log)
        return False
