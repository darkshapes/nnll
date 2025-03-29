#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import networkx as nx
from nnll_01 import debug_monitor


@debug_monitor
def build_conversion_graph():
    """create coordinate pair from valid conversions then deploy as a graph"""
    from nnll_15 import VALID_CONVERSIONS

    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(VALID_CONVERSIONS)

    return new_graph


@debug_monitor
def label_edge_attrib_for(nx_graph: nx.Graph, ollama: bool = False, hf_hub: bool = False) -> nx.Graph:
    """
    Build graph and assign edge attributes to it\n
    :param nx_graph: Preassembled graph of models to label
    :param ollama: Whether to process ollama models
    :param hf_hub: Whether to process hub models
    :return: A graph with model data attached to it
    """
    import os

    if ollama:
        from nnll_15 import from_ollama_cache

        ollama_models = from_ollama_cache()
        for model in ollama_models:
            nx_graph.add_edges_from(
                model.available_tasks,
                entry=model,
                key=os.path.basename(model.model),
                library=model.library,
                model_id=model.model,
                size=model.size,
                time=model.timestamp,
                weight=1.0,
            )
    if hf_hub:
        from nnll_15 import from_hf_hub_cache

        hub_models = from_hf_hub_cache()
        for model in hub_models:
            nx_graph.add_edges_from(
                model.available_tasks,
                entry=model,
                library=model.library,
                model_id=model.model,
                name=os.path.basename(model.model),
                size=model.size,
                time=model.timestamp,
                weight=1.0,
            )
    return nx_graph


@debug_monitor
def trace_objective(nx_graph: nx.Graph, prompt_type: str, target: str):
    """
    Find a valid path from current state (prompt_type) to designated state (target)\n
    :param nx_graph: Model graph to use for tracking operation order
    :param prompt_type: Input prompt state/states
    :param target: The user-seleccted end-state
    :return: An iterator for the edges forming a way towards the target, or Note
    """

    # Ensure path exists (otherwise 'bidirectional' may loop infinitely)
    if nx.has_path(nx_graph, prompt_type, target):
        model_path = nx.bidirectional_shortest_path(nx_graph, prompt_type, target)
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
    return nx_graph, process_type, target
