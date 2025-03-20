# # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
# # <!-- // /*  d a r k s h a p e s */ -->
from textual import work
from textual.widgets import TextArea


class MessagePanel(TextArea):

    @work(exclusive=True)
    async def erase_message(self):
        """Empty panel contents"""
        self.clear()
