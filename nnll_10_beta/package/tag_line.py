#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


import networkx as nx
from textual import work
from textual.reactive import reactive
from nnll_10_beta.package.model_registry import VALID_CONVERSIONS  # , from_ollama_cache
from nnll_10_beta.package.conversion_graph import assign_edge_attributes

from nnll_10_beta.package.carousel import Carousel


class TagLine(Carousel):
    """Output media selection field"""

    valid_conversions: dict = VALID_CONVERSIONS
    current_output: reactive[tuple] = reactive(("", ""), recompose=True)
    nx_graph = ""

    def on_mount(self):
        self.current_output = next(iter(self.valid_conversions))
        self.add_columns(("0", "1"))
        self.add_rows([row.strip()] for row in self.valid_conversions)
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
        self.nx_graph = assign_edge_attributes()

    @work(exclusive=True)
    async def trace_model_path(self, in_type: str = "text", out_type: str = "text"):
        model_path = nx.bidirectional_shortest_path(self.nx_graph, "text", "text")
        first_hop = self.nx_graph[next(iter(model_path))]
