#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


import sys
import os

# pylint:disable=import-outside-toplevel
sys.path.append(os.getcwd())

import networkx as nx
from nnll_01 import debug_monitor, info_message as nfo
# from nnll_15.constants import ModeType


class IntentProcessor:
    intent_graph: nx.Graph = None
    coordinates_path: dict = None
    registry_entries: dict = None
    model_names: dict = None

    def __init__(self):
        """
        Create instance of graph processor & initialize objectives for tracing paths
        """

    @debug_monitor
    def calculate_intent_graph(self):
        """Generate and store the intent graph."""
        from nnll_14 import calculate_graph

        self.intent_graph = calculate_graph()
        return self.intent_graph

    @debug_monitor
    async def confirm_available_graph(self):
        if not self.intent_graph:
            raise ValueError("Intent graph not calculated.")

    @debug_monitor
    async def confirm_coordinates_path(self):
        if not self.coordinates_path:
            raise ValueError("Coordinates not plotted.")

    @debug_monitor
    async def confirm_model_waypoints(self):
        if not self.registry_entries:
            raise ValueError("Registry not populated.")

    @debug_monitor
    def derive_coordinates_path(self, mode_in: str, mode_out: str):
        """
        Derive the coordinates path based on traced objectives.
        :param prompt_type: If provided, will use this. Otherwise, resolves from content.
        """
        from nnll_14 import trace_objective

        self.confirm_available_graph()
        self.coordinates_path = trace_objective(self.intent_graph, mode_in=mode_in, mode_out=mode_out)

    @debug_monitor
    def define_model_waypoints(self):
        from nnll_05 import pull_path_entries

        self.confirm_available_graph()
        self.confirm_coordinates_path()
        self.registry_entries = pull_path_entries(self.intent_graph, self.coordinates_path)
        self.model_names = [x["entry"].model for x in list(self.registry_entries)]
        nfo(vars(self), dir(self))


# IntentPath = NamedTuple("IntentPath",[()])

# from nnll_05 import machine_intent, pull_path_entries, label_key_prompt
# intent_graph = calculate_graph()

# coordinates_path = pull_path_entries(intent_graph, trace_objective(intent_graph, prompt_type = label_key_prompt(content), target))
# yielde machine_intent(content[prompt_type], coordinates_path)

# intent_graph = calculate_graph()

# # Instance 2: Derive coordinates path from tracing and resolving prompt
# prompt_type = label_key_prompt(content)
# traced_path = trace_objective(intent_graph, prompt_type=prompt_type, target=target)
# coordinates_path = pull_path_entries(intent_graph, traced_path)

# # Final inference using machine_intent with the two instances
# result = machine_intent(content[prompt_type], coordinates_path)

#             # Run the computationally heavy inference step
