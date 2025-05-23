#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import networkx as nx
from nnll_01 import debug_monitor
from nnll_01 import nfo  # , dbug
from nnll_60 import CONFIG_PATH_NAMED, JSONCache
# from nnll_15.constants import LibType

mir_db = JSONCache(CONFIG_PATH_NAMED)


def lookup_function_for(known_repo: str, delimiter=".") -> str:
    """
    Find MIR URI from known repo name and retrieve its \n
    MIR data and call instructions autofilled by decorator\n
    :param known_repo: HuggingFace repo name
    :param mir_data: MIR URI reference file
    :param delimiter: The separator between module and function identities
    :return: `str` of the mir URI
    """
    import importlib
    from io import TextIOWrapper

    @mir_db.decorator
    def _read_data(data: TextIOWrapper = None):
        return data

    mir_data = _read_data()
    mir_arch = next(key for key, value in mir_data.items() if known_repo in value.get("repo"))
    map_entry = mir_data[mir_arch].get("constructor", "")  # pylint: disable=unsubscriptable-object
    # needs validation
    run_map = {"image": "nnll_64.run_inference", "speech": "", "text": "nnll_64.run_inference"}
    sequence = run_map.get(map_entry)
    nfo(f"lookup result : {mir_arch}, {sequence}")

    sequence = sequence.split(delimiter)
    module = importlib.import_module(sequence[0])
    constructor = getattr(module, sequence[-1])
    nfo(f"attr {constructor}, {mir_arch}")
    return constructor, mir_arch


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


# lookup function for model file from registry
# import os
# if os.path.isfile(known_repo):
