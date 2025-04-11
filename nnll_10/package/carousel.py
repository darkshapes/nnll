#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Selection Function"""

from textual.widgets import DataTable
from textual.screen import Screen
from textual.reactive import reactive

from nnll_01 import debug_monitor


class Carousel(DataTable):
    """Revolving text-line component based on default DataTable widget class"""

    nx_graph: dict
    content_cell: reactive[int] = reactive(0.0)

    up = "[@click='scroll_button(1)']▲[/]"
    dwn = "[@click='scroll_button']▼[/]"

    def on_mount(self) -> None:
        self.show_header = False
        self.cursor_type = "cell"

    @debug_monitor
    def emulate_scroll_down(self, interval: float = 0.05) -> str:
        """Trigger datatable cursor movement using fractional sensitivity
        :param ceiling: Total entry count of the table *column*
        :return: The datata in the table *row*
        """
        if self.content_cell < len(self.columns) - 1:
            self.content_cell += interval
        coordinate = abs(int(round(self.content_cell)))
        self.move_cursor(row=coordinate, column=1)
        current_content = self.get_cell_at((coordinate, 1))
        return current_content

    @debug_monitor
    def emulate_scroll_up(self, interval: float = 0.05) -> str:
        """Trigger datatable cursor movement using fractional sensitivity
        :param ceiling: Total entry count of the table *column*
        :return: The datata in the table *row*
        """
        if self.content_cell > 0:
            self.content_cell -= interval
        coordinate = abs(int(round(self.content_cell)))
        self.move_cursor(row=coordinate, column=1)
        current_content = self.get_cell_at((coordinate, 1))
        return current_content

    @debug_monitor
    def action_scroll_button(self, up: bool = False) -> None:
        """Manually trigger scrolling and panel switching
        :param up: Scroll direction, defaults to False
        """
        if up:
            content = self.emulate_scroll_up(interval=0.6)
            if self.id == "input_tag":
                self.query_ancestor(Screen).foldr["ps"].current = self.query_ancestor(Screen).input_map[content]
        else:
            content = self.emulate_scroll_down(interval=0.6)
            if self.id == "input_tag":
                self.query_ancestor(Screen).foldr["ps"].current = self.query_ancestor(Screen).input_map[content]
