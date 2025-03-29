#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
from typing import Any
import networkx as nx
from nnll_01 import debug_monitor, info_message as nfo
from nnll_11 import chat_machine
from nnll_60 import JSONCache, CONFIG_PATH_NAMED
from nnll_14 import trace_objective  # , loop_in_feature_processes

mir_db = JSONCache(CONFIG_PATH_NAMED)


@debug_monitor
def resolve_prompt(content: Any = None) -> tuple[str, nx.Graph]:
    """Assess prompt data streams, output primary source input\n
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
    Find MIR id from known repo name and run generation\n
    MIR data and call instructions autofilled by decorator\n
    :param known_repo: HuggingFace repo name
    :param mir_data: MIR URI reference file
    :return: `str` of the mir URI
    """

    mir_arch = next(key for key, value in mir_data() if known_repo in value["repo"])
    sequence = mir_data[mir_arch].get("constructor")
    call_sequence = [split_sequence_by(".", seq) for seq in sequence]
    module_names = [function[0] for function in call_sequence]
    function_names = [function[-1] for function in call_sequence]
    operations = zip(module_names, function_names)
    return operations


@debug_monitor
def execute_path(nx_graph: nx.Graph, traced_path: list[tuple], prompt: Any) -> None:
    """Create operating instructions from user input"""
    import importlib

    output_map = {0: prompt}
    for i in range(len(traced_path) - 1):
        nfo(nx_graph["text"])
        registry_entry = nx_graph[traced_path[i]][traced_path[i + 1]]
        nfo(registry_entry)
        current_entry = registry_entry[next(iter(registry_entry))]
        current_model = current_entry.get("model_id")
        nfo(f"current model : {current_model}")
        if current_entry.get("library") == "hub":
            operations = lookup_function_for(current_model)
            import_name = next(iter(operations))
            module = importlib.import_module(import_name)
            func = getattr(module, module[import_name])
            new_output = func(current_model, output_map[i])
            output_map.setdefault(i + 1, new_output)

        elif current_entry.get("library") == "ollama":
            new_output = chat_machine(current_model, output_map[i])
            output_map.setdefault(i + 1, new_output)
    return output_map


def main(nx_graph: nx.Graph, content: dict, target: str):
    prompt_type = resolve_prompt(content)  # , aux_processes
    traced_path = trace_objective(nx_graph, prompt_type, target)
    if traced_path is not None:
        # if len(aux_processes) > 0:
        #     for process_type in aux_processes:  # temporarily add attribute to nx_graph
        #         nx_graph = loop_in_feature_processes(nx_graph, prompt_type, target)
        response = execute_path(nx_graph, traced_path, content[prompt_type])
        return response
