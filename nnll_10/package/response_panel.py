#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work

# from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import TextArea

# from nnll_01 import debug_monitor
from nnll_10.package.chat_machine import chat_machine
# from nnll_14 import build_conversion_graph, label_edge_attrib_for
# from nnll_05 import hf_repo_to_mir_arch


class ResponsePanel(TextArea):
    """Machine response field"""

    # nx_graph = None
    # target_options = (,)
    prefix: str = "ollama_chat/"
    is_generating: reactive[bool] = reactive(False)

    def on_mount(self):
        # nx_graph = build_conversion_graph()
        # self.nx_graph = label_edge_attrib_for(nx_graph, 1,1)
        # self.target_options = set([edge[1] for edge in nx_graph.edges]) # as in : ('text','text')[1]
        self.language = "markdown"
        self.read_only = True
        self.soft_wrap = True

    @work(group="chat")
    async def generate_response(self, model, message):  # content, target):
        """Fill display with generated content"""
        self.is_generating = True
        # output = execute_model_path(input_types: list[str], content[image|text|np.array], target)
        # return output
        #
        # @work(group="chat")
        # async def generate_from_ollama(model_id, prompt)
        self.move_cursor(self.document.end)
        self.scroll_end(animate=True)
        self.insert("\n---\n")
        model = f"{self.prefix}{model}"
        self.move_cursor(self.document.end, center=True)
        # adjust for multimodality of vision/text
        try:
            async for chunk in chat_machine(model, message):
                self.insert(chunk)
                self.scroll_cursor_visible(center=True, animate=True)
        except AttributeError as error_log:
            print(error_log)
        # self.is_generating = False
        # return output
        return False

    # @work(group="chat")
    # async def trace_model_path(self, source: str = "text", target: str = "text", prompt) -> :
    # convert source data into str type
    # convert target data into str type
    # model_path = path_objective(source,target)
    # output = False
    # if model_path:
    # async for i in range(len(model_path) - 1):
    #      prompt = output if output else prompt
    #     next_model = nx_graph[model_path[i]][model_path[i+1]]
    #     if next_model[next(iter(next_model))].get('library') == 'hub'
    #        output = generate_from_hf_repo(next_model.model_id, prompt)
    #     else:
    #        output = generate_from_ollama(next_model.model_id, prompt)
    # return output
