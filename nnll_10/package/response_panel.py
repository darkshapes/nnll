#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


from textual import work
from textual.reactive import reactive
from textual.widgets import TextArea
# import networkx as nx

from nnll_01 import debug_message as dbug
from nnll_05 import main
from nnll_11 import chat_machine


class ResponsePanel(TextArea):
    """Machine response field"""

    is_generating: reactive[bool] = reactive(False)

    def on_mount(self) -> None:
        self.language = "markdown"

        self.read_only = True
        self.soft_wrap = True

    @work(group="chat")
    async def scribe_response(self, nx_graph: dict, message: dict[str], target: str) -> None:
        """Write a text response to current widget"""

        self.is_generating = True
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")

        self.move_cursor(self.document.end, center=True)

        try:
            async for chunk in main(nx_graph=nx_graph, content=message, target=target):
                # todo : allow user selection between cursor jumps
                self.move_cursor(self.document.end)
                self.scroll_cursor_visible(center=True, animate=True)
                self.scroll_end(animate=True)
                self.insert(chunk)
        except GeneratorExit as error_log:
            dbug(error_log)
        except RuntimeError as error_log:
            dbug(error_log)
        except AttributeError as error_log:
            dbug(error_log)
        self.is_generating = False
