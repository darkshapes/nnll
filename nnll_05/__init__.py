#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Any
import networkx as nx
from nnll_01 import debug_monitor, info_message as nfo

# from nnll_11 import
from nnll_60 import JSONCache, CONFIG_PATH_NAMED
from nnll_14 import trace_objective  # , loop_in_feature_processes

mir_db = JSONCache(CONFIG_PATH_NAMED)


@debug_monitor
def resolve_prompt(content: Any = None) -> tuple[str, nx.Graph]:
    """Distil multi-prompt input streams into a single primary source\n
    Use secondary streams for supplementary processes.\n
    **TL;DR**: text prompts take precedence when available\n
    :param content: User-supplied input data
    :param target: User-supplied objective data state
    :return: `str` Initial conversion state\n\n
    ```
    dict       medium   data
            ,-text     str|dict
            '-image    array
    content-'-speech   array
            '-video    array
            '-music    array
    ```
    """
    prompt_type = None
    aux_processes = []
    if not content.get("text", 0) or len(content.get("text")) == 0:
        for medium, data in content.items():
            if data and len(data) > 1:
                prompt_type = str(medium)
    else:
        prompt_type = "text"
        test_content = content.copy()
        test_content.pop("text")
        for medium, data in test_content.items():
            if data and len(data) > 1:
                print(f"medium : {medium}")
                aux_processes.append(medium)
    return prompt_type  # , aux_processes


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
def lookup_function_for(known_repo: str, mir_data: dict = None) -> str:
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
    if nx.has_path(nx_graph, traced_path[0], traced_path[1]):
        registry_entries = [  # ruff : noqa
            nx_graph[traced_path[i]][traced_path[i + 1]][hop]  #
            for i in range(len(traced_path) - 1)  #
            for hop in nx_graph[traced_path[i]][traced_path[i + 1]]  #
        ]
    return registry_entries


def execute_path(traced_path: list[tuple], prompt: Any, registry_entries: dict) -> None:
    """Execute on instructions selected previously"""

    output_map = {0: prompt}
    for i in range(len(traced_path) - 1):
        current_entry = registry_entries[next(iter(registry_entries))]
        current_model = current_entry.get("model_id")
        current_library = current_entry.get("library")
        nfo(f"current model : {current_model}")
        if current_library == "hub":
            function_call = lookup_function_for(current_model)
            new_output = function_call(current_model, output_map[i])
            output_map.setdefault(i + 1, new_output)
        elif current_library == "ollama":
            new_output = chat_machine(current_model, current_library, output_map[i])
            output_map.setdefault(i + 1, new_output)
        elif current_library == "lmstudio":
            new_output = chat_machine(current_model, current_library, output_map[i])
            output_map.setdefault(i + 1, new_output)
    return output_map


def main(nx_graph: nx.Graph, content: dict, target: str):
    prompt_type = resolve_prompt(content)  # , aux_processes
    traced_path = trace_objective(nx_graph, prompt_type, target)
    if traced_path is not None:
        registry_entries = pull_path_entries(nx_graph, traced_path)
        # if len(aux_processes) > 0:
        #     for process_type in aux_processes:  # temporarily add attribute to nx_graph
        #         nx_graph = loop_in_feature_processes(nx_graph, prompt_type, target)
        response = execute_path(traced_path, content[prompt_type], registry_entries)
        return response
