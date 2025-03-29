#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->
import asyncio

from textual import work
from textual.reactive import reactive
from textual.widgets import TextArea
# import networkx as nx

# from nnll_05 import lookup_function_for, loop_in_feature_processes, resolve_prompt
from nnll_10.package.chat_machine import chat_machine
# from nnll_14 import build_conversion_graph, label_edge_attrib_for, trace_objective


class ResponsePanel(TextArea):
    """Machine response field"""

    # nx_graph: Dict = None
    # target_options: tuple = ()
    prefix: str = "ollama_chat/"
    is_generating: reactive[bool] = reactive(False)

    def on_mount(self) -> None:
        self.language = "markdown"
        # nx_graph = build_conversion_graph()
        # self.nx_graph = label_edge_attrib_for(nx_graph, 1, 1)
        # self.target_options = set([edge[1] for edge in nx_graph.edges])  # endpoints, as in : (_,'text')[1]
        self.read_only = True
        self.soft_wrap = True

    @work(group="chat")
    async def generate_response(self, model: str, message: dict[str]):  # , content, target):  #
        """Signal start of generation content given user input"""
        self.is_generating = True
        model = f"{self.prefix}{model}"

        asyncio.gather(self.scribe_response(model, message))
        self.is_generating = False
        return False

    @work(group="chat")
    async def scribe_response(self, model: str, message: dict[str]) -> None:
        """Write a text response to current widget"""
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")
        self.move_cursor(self.document.end, center=True)
        try:
            async for chunk in chat_machine(model, message):
                # todo : allow user selection between cursor jumps
                self.move_cursor(self.document.end)
                self.scroll_cursor_visible(center=True, animate=True)
                self.scroll_end(animate=True)
                self.insert(chunk)
        except AttributeError as error_log:
            print(error_log)
