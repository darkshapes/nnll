#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Selection Function"""

from textual.widgets import DataTable
from textual.reactive import reactive

from nnll_01 import debug_monitor


class Carousel(DataTable):
    nx_graph: dict
    content_cell: reactive[int] = reactive(0.0)

    up = "▲ "
    dwn = " ▼"

    def on_mount(self) -> None:
        self.show_header = False
        self.cursor_type = "cell"

    @debug_monitor
    def emulate_scroll_down(self, ceiling: int) -> str:
        """Trigger datatable cursor movement with cumulative fraction
        :param ceiling: Total entry count of the table *column*
        :return: The datata in the table *row*
        """
        if self.content_cell < ceiling - 1:
            self.content_cell += 0.05
        current_content = self.tag_line_scroller(self.content_cell)
        return current_content

    @debug_monitor
    def emulate_scroll_up(self) -> str:
        """Trigger datatable cursor movement with cumulative fraction
        :param ceiling: Total entry count of the table *column*
        :return: The datata in the table *row*
        """
        if self.content_cell > 0:
            self.content_cell -= 0.05
        current_content = self.tag_line_scroller(self.content_cell)
        return current_content

    @debug_monitor
    def tag_line_scroller(self, content_cell) -> str:
        """Increment or decrement current datatable position
        Move cursor to next data table entry as if scrolling
        Action ready when `round` of scrolls reaches threshold
        :return: The next iteration of the table"""
        coordinate = int(round(content_cell))
        self.move_cursor(row=coordinate, column=1)
        current_content = self.get_cell_at((coordinate, 1))
        return current_content
