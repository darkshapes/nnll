#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Dict
import networkx as nx
from nnll_01 import debug_monitor


@debug_monitor
def build_conversion_graph():
    """create coordinate pair from valid conversions then deploy as a graph"""
    import networkx as nx
    from nnll_15 import VALID_CONVERSIONS

    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(VALID_CONVERSIONS)

    return new_graph


@debug_monitor
def assign_edge_attributes(nx_graph):
    """Build graph and assign edge attributes to it"""
    import os
    from nnll_15 import from_ollama_cache, from_hf_hub_cache

    ollama_models = from_ollama_cache()
    hub_models = from_hf_hub_cache()
    for model in ollama_models:
        print(model.available_tasks)
        nx_graph.add_edges_from(model.available_tasks, key=os.path.basename(model.model), model_id=model.model, size=model.size, weight=1.0)
    for model in hub_models:
        nx_graph.add_edges_from(model.available_tasks, key=os.path.basename(model.model), model_id=model.model, size=model.size, weight=1.0)
    return nx_graph


@debug_monitor
def path_objective(nx_graph: nx.Graph, source: str, target: str) -> Dict:
    import networkx as nx

    path = nx.bidirectional_shortest_path(nx_graph, source, target)
