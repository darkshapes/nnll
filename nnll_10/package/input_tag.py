#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual.reactive import reactive
from textual.screen import Screen

from nnll_10.package.carousel import Carousel


class InputTag(Carousel):
    """Populate Input Types List"""

    target_options: reactive[set] = reactive({})

    def on_mount(self):
        nx_graph = self.query_ancestor(Screen).nx_graph
        # note: Length sort on target_options kinda poor way to get text on top, works for the moment
        self.target_options = sorted({edge[1] for edge in nx_graph.edges}, key=len)
        self.add_columns("0", "1", "2")
        self.add_rows([self.up, row.strip(), self.dwn] for row in self.target_options)
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
