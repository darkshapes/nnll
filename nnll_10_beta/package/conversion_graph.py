#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
import networkx as nx
from nnll_10_beta.package.model_registry import from_ollama_cache, from_hf_hub_cache, VALID_CONVERSIONS


def build_conversion_graph():
    """create coordinate pair from valid conversions then deploy as a graph"""
    new_graph = nx.MultiDiGraph()
    new_graph.add_nodes_from(VALID_CONVERSIONS)

    return new_graph


def assign_edge_attributes():
    """Build graph and assign edge attributes to it"""
    nx_graph = build_conversion_graph()
    ollama_models = from_ollama_cache()
    hub_models = from_hf_hub_cache()
    for model, details in ollama_models.items():
        nx_graph.add_edges_from(details.available_tasks, key=os.path.basename(model), model_id=model, size=details.size, weight=1.0)
    for model, details in hub_models.items():
        nx_graph.add_edges_from(details.available_tasks, key=os.path.basename(model), model_id=model, size=details.size, weight=1.0)

    return nx_graph
