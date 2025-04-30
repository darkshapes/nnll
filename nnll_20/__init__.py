#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


from textual import work
from textual.reactive import reactive
from textual.widgets import TextArea
import networkx as nx

from nnll_01 import dbug
# from nnll_11 import chat_machine


class ResponsePanel(TextArea):
    """Machine response field"""

    nx_graph: nx.Graph = None
    target_options: tuple = ()
    is_generating: reactive[bool] = reactive(False)

    def on_mount(self) -> None:
        # self.language = "markdown"
        self.read_only = True
        self.soft_wrap = True

    @work(group="chat")
    async def scribe_response(self, model: str, message: dict[str]) -> None:
        """Write a text response to current widget"""
        from nnll_15 import LibType

        self.is_generating = True
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")

        self.move_cursor(self.document.end, center=True)

        try:
            async for chunk in chat_machine(library=LibType.OLLAMA, message=message, model=model):
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
