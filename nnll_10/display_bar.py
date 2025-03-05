#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import tiktoken

# from textual.reactive import reactive, var
from textual.widgets import DataTable
from textual.color import Color


class DisplayBar(DataTable):
    """Thin mid-screen display/interaction"""

    CLASS_PATH = "combo.tcss"
    UNIT1 = "chr /"  # Display Bar Units
    UNIT2 = "tkn /"
    rows: list[tuple] = [
        (0, 0),
        (f"      {UNIT1}   ", f" {UNIT2}   "),
    ]

    # count: reactive[int] = var(0)  # Offset for sparkline scrolling & typing metric
    # self.count += 1

    def on_mount(self) -> None:
        self.styles.background = Color(0, 0, 0)
        self.styles.back = Color(0, 0, 0)
        self.styles.color = "#333333"
        self.show_row_labels = False
        self.show_header = False
        self.cursor_type = None

    async def calculate_tokens(self, message):  # field: TextArea.Changed):
        """Live display of tokens and characters"""
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(message))
        character_count = len(message)
        self.update_cell_at((0, 0), f"     {character_count}{self.UNIT1}")
        self.update_cell_at((0, 1), f"{token_count}{self.UNIT2}")
