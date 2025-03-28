#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Dict
import networkx as nx
from nnll_01 import debug_monitor
from nnll_15 import RegistryEntry


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
def path_objective(nx_graph: nx.Graph, source: str, target: str):
    """
    Find a valid path from current state (source) to designated state (target)\n
    :param nx_graph: Model graph to use for tracking operation order
    :param source: Input prompt state/states
    :param target: The user-seleccted end-state
    :return: An iterator for the edges forming a way towards the target, or Note
    """

    # Ensure path exists (otherwise 'bidirectional' loops infinitely)
    if nx.has_path(nx_graph, source, target):
        model_path = nx.bidirectional_shortest_path(nx_graph, source, target)
        return model_path


# get all neighbor connection
# nx_graph['speech']['text']

# get all model name on graph
# nx_graph.edges.data('keys')

# change attribute
# nx_graph.edges[‘text’, ‘image’][‘weight'] = 4.2
# edge_attrib nx.get_edge_attributes(nx_graph,'key')

# get number of edges/paths directed away
# nx_graph.out_degree('text')

# get number of edges/paths directed towards
# nx_graph.in_degree('text')

# show all edge attributes by index
# node_attrib nx.get_edge_attributes(nx_graph,'key')

# seen2 = set([e[1] for e in nx_graph.edges]) # list all potential target states (to populate list)
