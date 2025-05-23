# # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
# # <!-- // /*  d a r k s h a p e s */ -->
from textual import work
from textual.widgets import TextArea

# from nnll_01 import debug_monitor


class MessagePanel(TextArea):
    def on_mount(self):
        self.cursor_blink = False

    @work(exclusive=True)
    async def erase_message(self):
        """Empty panel contents"""
        self.clear()
