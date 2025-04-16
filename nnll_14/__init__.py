#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

# from typing import NamedTuple
import networkx as nx
from nnll_01 import debug_monitor, info_message as nfo  # , debug_message as dbug


@debug_monitor
def calculate_graph() -> nx.Graph:
    """Create coordinate pairs from valid conversions then deploy as a graph
     assign edge attributes\n
    :param nx_graph: Preassembled graph of models to label
    :param ollama: Whether to process ollama models
    :param hf_hub: Whether to process hub models
    :return: Graph modeling all current ML/AI tasks appended with model data"""
    from nnll_15 import VALID_CONVERSIONS, from_cache

    nx_graph = nx.MultiDiGraph()
    nx_graph.add_nodes_from(VALID_CONVERSIONS)
    registry_entries = from_cache()
    # nfo(registry_entries)
    if registry_entries:
        for model in registry_entries:
            nx_graph.add_edges_from(model.available_tasks, entry=model, weight=1.0)
        return nx_graph
    else:
        nfo("Registry error, graph attributes not applied")
        return nx_graph


@debug_monitor
def trace_objective(nx_graph: nx.Graph, mode_in: str, mode_out: str):
    """Find a valid path from current state (mode_in) to designated state (mode_out)\n
    :param nx_graph: Model graph to use for tracking operation order
    :param mode_in: Input prompt type or starting state/states
    :param mode_out: The user-selected ending-state
    :return: An iterator for the edges forming a way towards the mode out, or Note"""

    model_path = None
    if nx.has_path(nx_graph, mode_in, mode_out):  # Ensure path exists (otherwise 'bidirectional' may loop infinitely)
        if mode_in == mode_out and mode_in != "text":
            orig_mode_out = mode_out  # Solve case of non-text self-loop edge being incomplete transformation
            mode_out = "text"
            model_path = nx.bidirectional_shortest_path(nx_graph, mode_in, mode_out)
            model_path.append(orig_mode_out)
        else:
            model_path = nx.bidirectional_shortest_path(nx_graph, mode_in, mode_out)
            if len(model_path) == 1:
                model_path.append(mode_out)  # this behaviour likely to change in future
    return model_path
