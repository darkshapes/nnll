#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work
from textual.widgets import TextArea


class ReadOut(TextArea):  # machine output
    """Display window for text"""

    @work(exit_on_error=False)
    async def prepare_display_log(self):
        """Ready display for new text"""
        self.insert("---\n")
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.line_number_start = 3
        return self
