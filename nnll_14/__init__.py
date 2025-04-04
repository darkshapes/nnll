#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import networkx as nx
from nnll_01 import debug_monitor
from nnll_15.constants import LibType


@debug_monitor
def build_conversion_graph():
    """create coordinate pair from valid conversions then deploy as a graph"""
    from nnll_15 import VALID_CONVERSIONS

    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(VALID_CONVERSIONS)

    return new_graph


@debug_monitor
def label_edge_attrib_for(nx_graph: nx.Graph, lib_type: LibType) -> nx.Graph:
    """
    Build graph and assign edge attributes to it\n
    :param nx_graph: Preassembled graph of models to label
    :param ollama: Whether to process ollama models
    :param hf_hub: Whether to process hub models
    :return: A graph with model data attached to it
    """
    from nnll_15 import RegistryEntry

    registry_entries = RegistryEntry.from_model_data(lib_type)
    for model in registry_entries:
        nx_graph.add_edges_from(model.available_tasks, entry=model, weight=1.0)
    return nx_graph


@debug_monitor
def trace_objective(nx_graph: nx.Graph, source: str, target: str):
    """
    Find a valid path from current state (source) to designated state (target)\n
    :param nx_graph: Model graph to use for tracking operation order
    :param source: Input prompt type or starting state/states
    :param target: The user-selected ending-state
    :return: An iterator for the edges forming a way towards the target, or Note
    """

    model_path = None
    if nx.has_path(nx_graph, source, target):  # Ensure path exists (otherwise 'bidirectional' may loop infinitely)
        if source == target and source != "text":
            original_target = target  # Solve case of non-text self-loop edge being incomplete transformation
            target = "text"
            model_path = nx.bidirectional_shortest_path(nx_graph, source, target)
            model_path.append(original_target)
        else:
            model_path = nx.bidirectional_shortest_path(nx_graph, source, target)
            if len(model_path) == 1:
                model_path.append(target)
    return model_path


@debug_monitor
def loop_in_feature_processes(nx_graph: nx.Graph, process_type: str, target: str) -> nx.Graph:
    """loop in additional process chains based on target
    graft to init/callback of imported function
    add processes to appropriate graph edges

    feature switches available to user:
        none: (for single-input streams)
        description (t:s:i:v:m:meta analysis)
        reference (t:s:i:v:m:retrieval augmentation)
        identity (t:writer/i:face/s:voice/v:m:phrasing))
        language (t/s:interpret/i:pose/v:gesture/m:genre+instrumentation))
    """
    # nx_graph.edges[‘text’, ‘image’]['aux'] = [process_names,more_process_names]

    return nx_graph
