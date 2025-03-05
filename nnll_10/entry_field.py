#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work, on

# from textual.app import ComposeResult
from textual.widgets import TextArea


class EntryField(TextArea):  # machine input
    """Large input field for text entry"""

    @work(exit_on_error=False)
    async def read_text_field(self):
        """Return all the text in the widget"""
        message = self.text
        return message

    @on(TextArea.Changed)
    async def update_tokens(self):
        return self.read_text_field()
