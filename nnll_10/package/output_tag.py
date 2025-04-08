#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual.reactive import reactive
from textual.screen import Screen

from nnll_10.package.carousel import Carousel


class OutputTag(Carousel):
    """Tag for showing output method"""

    nx_graph: dict
    target_options: reactive[dict] = reactive({})

    def on_mount(self):
        self.nx_graph = self.query_ancestor(Screen).nx_graph
        self.target_options = {edge[0] for edge in self.nx_graph.edges}
        self.add_columns(("0", "1"))
        self.add_rows([row.strip()] for row in self.target_options)
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
