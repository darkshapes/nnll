#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Any
import networkx as nx
from nnll_01 import debug_monitor
from nnll_01 import info_message as nfo
from nnll_60 import CONFIG_PATH_NAMED, JSONCache
from nnll_15.constants import LibType

mir_db = JSONCache(CONFIG_PATH_NAMED)


@debug_monitor
def split_sequence_by(delimiter: str = ".", sequence: str = "") -> tuple | None:
    """Divide a string into an import module and a function
    :param delimiter: The separator between component identities
    :param sequence: The string to split
    :return: `tuple` of import and function statement
    """
    parts = sequence.rsplit(delimiter, 1)
    return parts[0], parts[1] if len(parts) > 1 else None


@mir_db.decorator
async def lookup_function_for(known_repo: str, mir_data: dict = None) -> str:
    """
    Find MIR URI from known repo name and retrieve its \n
    MIR data and call instructions autofilled by decorator\n
    :param known_repo: HuggingFace repo name
    :param mir_data: MIR URI reference file
    :return: `str` of the mir URI
    """
    import importlib

    mir_arch = next(key for key, value in mir_data() if known_repo in value["repo"])
    sequence = mir_data[mir_arch].get("constructor", [])
    call_sequence = [seq.split(".") for seq in sequence]
    module_names, function_names = zip(*[(func[0], func[-1]) for func in call_sequence]) if call_sequence else ([], [])
    modules = zip(module_names, function_names)
    import_name = next(iter(modules))
    module = importlib.import_module(modules)
    function_call = getattr(module, module[import_name])
    return function_call


@debug_monitor
def pull_path_entries(nx_graph: nx.Graph, traced_path: list[tuple]) -> None:
    """Create operating instructions from user input
    Trace the next hop along the path, collect all compatible models
    Set current model based on weight and next available
    """
    registry_entries = []
    if traced_path is not None and nx.has_path(nx_graph, traced_path[0], traced_path[1]):
        registry_entries = [  # ruff : noqa
            nx_graph[traced_path[i]][traced_path[i + 1]][hop]  #
            for i in range(len(traced_path) - 1)  #
            for hop in nx_graph[traced_path[i]][traced_path[i + 1]]  #
        ]
    return registry_entries


# @debug_monitor
# def label_key_prompt(message: dict, mode_in: str) -> str:
#     """Distil multi-prompt input streams into a single primary source\n
#     Use secondary streams for supplementary processes.\n
#     **TL;DR**: text prompts take precedence when available\n
#     :param message: User-supplied input data
#     :param source: User-supplied origin data state
#     :return: `str` Initial conversion state\n\n
#     ```
#     dict       medium   data
#             ,-text     str|dict
#             '-image    array
#     message-'-speech   array
#             '-video    array
#             '-music    array
#     ```
#     """
#     if not mode_in:
#         if not message["text"]:
#             mode_in = next(iter([mode_in for mode_in in message if mode_in is not None and len(mode_in) > 1]))
#         else:
#             mode_in = "text"

#     return message.get(mode_in)


# async def machine_intent(message: Any, registry_entries: dict, coordinates_path: dict) -> Any:
#     """Execute on instructions selected previously"""
#     from nnll_11 import chat

#     label_prompt = label_key_prompt(message, mode_in=next(iter(coordinates_path)))
#     output_map = [label_prompt]
#     mode_hops = len(coordinates_path) - 1
#     nfo(mode_hops)
#     for i in range(mode_hops):
#         current_coords = next(iter(registry_entries)).get("entry")
#         nfo(f"current model : {current_coords}")
#         current_model = current_coords.model
#         current_library = current_coords.library
#         nfo(f"current model : {current_model}")
#         if current_library == LibType.HUB:
#             args = (current_model, output_map[i])
#             nfo(f"hub {current_model, current_library, output_map}")
#             output_map.append(((lookup_function_for(current_model), args)))
#         elif current_library == LibType.OLLAMA or current_library == LibType.LM_STUDIO or current_library == LibType.VLLM:
#             nfo(f"chat {current_model, current_library, output_map}")
#             output_map.append((chat.forward, {"message": output_map[i], "model": current_model, "library": current_library, "max_workers": 8}))
#     return output_map


# yield chat_machine(model=current_model, message=output_map[i], library=current_library)

# rxaligned
# self.traced_path = trace_objective(nx_graph=self.nx_graph, mode_in=in_type, mode_out=out_type)
# self.registry_entries = pull_path_entries(self.nx_graph, self.traced_path
# if len(aux_processes) > 0:
#     for process_type in aux_processes:  # temporarily add attribute to nx_graph
#         nx_graph = loop_in_feature_processes(nx_graph, prompt_type, mode_out)
