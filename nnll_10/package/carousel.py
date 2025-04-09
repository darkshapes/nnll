#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Selection Function"""

from textual.widgets import DataTable
from textual.reactive import reactive

from nnll_01 import debug_monitor


class Carousel(DataTable):
    nx_graph: dict
    content_cell: reactive[int] = reactive(0.0)
    arrow_up = "▲ "
    arrow_down = " ▼"

    def on_mount(self) -> None:
        self.show_header = False
        self.cursor_type = "cell"

    @debug_monitor
    def emulate_scroll_down(self, ceiling: list) -> str:
        """Trigger Datatable cursor movement down
        :param ceiling: Length of the information populating the table
        :return: The next iteration of the table"""

        if self.content_cell < ceiling - 1:
            self.content_cell += 0.1
        current_content = self.tag_line_scroller()
        return current_content

    @debug_monitor
    def emulate_scroll_up(self) -> str:
        """Trigger datatable cursor movement up
        :param content: The information populating the table
        :return: The next iteration of the table"""

        if self.content_cell > 0:
            self.content_cell -= 0.1
        current_content = self.tag_line_scroller()
        return current_content

    @debug_monitor
    def tag_line_scroller(self) -> str:
        """Increment or decrement current datatable position
        Only increments every `round` scroll
        :return: The next iteration of the table"""
        coordinate = int(round(self.content_cell))
        self.move_cursor(row=coordinate, column=0)
        # key_name = self.get_cell_at((coordinate, 0))
        current_content = self.get_cell_at((coordinate, 0))  # content.get(key_name)
        return current_content
