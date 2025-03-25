#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work

# from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import TextArea

# from nnll_01 import debug_monitor
from nnll_10.package.chat_machine import chat_machine
# from nnll_14 import build_conversion_graph, label_edge_attrib_for


class ResponsePanel(TextArea):
    """Machine response field"""

    # nx_graph = None
    prefix: str = "ollama_chat/"
    is_generating: reactive[bool] = reactive(False)

    def on_mount(self):
        # nx_graph = build_conversion_graph()
        # self.nx_graph = label_edge_attrib_for(nx_graph, 1,1)
        self.language = "markdown"
        self.read_only = True
        self.soft_wrap = True

    @work(group="chat")
    async def generate_response(self, model, message):
        """Fill display with generated content"""
        self.is_generating = True
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")
        model = f"{self.prefix}{model}"
        self.move_cursor(self.document.end, center=True)
        # run graph path routine here
        try:
            async for chunk in chat_machine(model, message):
                self.insert(chunk)
                self.scroll_cursor_visible(center=True, animate=True)
        except AttributeError as error_log:
            print(error_log)
        return False
