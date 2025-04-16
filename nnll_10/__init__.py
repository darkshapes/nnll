#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


import sys
import os
from typing import Callable


# pylint:disable=import-outside-toplevel
sys.path.append(os.getcwd())

import networkx as nx
from nnll_01 import debug_monitor, info_message as nfo

# from nnll_15.constants import ModeType
from nnll_15 import RegistryEntry


class IntentProcessor:
    intent_graph: nx.Graph = None
    coordinates_path: list[str] = None
    registry_entries: list[dict[RegistryEntry]] = None
    model_names: list[tuple[str]] = None
    # additional_model_names: dict = None

    def __init__(self):
        """
        Create instance of graph processor & initialize objectives for tracing paths
        """

    @debug_monitor
    def calculate_intent_graph(self) -> None:
        """Generate and store the intent graph."""
        from nnll_14 import calculate_graph

        self.intent_graph = calculate_graph()
        return self.intent_graph

    @debug_monitor
    async def confirm_available_graph(self) -> None:
        if not self.intent_graph:
            raise ValueError("Intent graph not calculated.")

    @debug_monitor
    async def confirm_coordinates_path(self) -> None:
        if not self.coordinates_path:
            raise ValueError("Coordinates not plotted.")

    @debug_monitor
    async def confirm_model_waypoints(self) -> None:
        if not self.registry_entries:
            raise ValueError("Registry not populated.")

    @debug_monitor
    def derive_coordinates_path(self, mode_in: str, mode_out: str, define=True) -> None:
        """
        Derive the coordinates path based on traced objectives.
        :param prompt_type: If provided, will use this. Otherwise, resolves from content.
        """
        from nnll_14 import trace_objective

        self.confirm_available_graph()
        self.coordinates_path = trace_objective(self.intent_graph, mode_in=mode_in, mode_out=mode_out)
        if define:
            self.define_model_waypoints()

    @debug_monitor
    def define_model_waypoints(self) -> None:
        from nnll_05 import pull_path_entries

        self.confirm_available_graph()
        self.confirm_coordinates_path()
        self.registry_entries = pull_path_entries(self.intent_graph, self.coordinates_path)
        self.registry_entries = sorted(self.registry_entries, key=lambda x: x["weight"])
        self.model_names = []
        temp_model_names = []
        for entry in self.registry_entries:
            model_name = entry["entry"].model
            if int(entry.get("weight")) == 0:
                self.model_names.append((f"*{os.path.basename(model_name)}", model_name))
            else:
                temp_model_names.append((os.path.basename(model_name), model_name))
        self.model_names.extend(temp_model_names)
        nfo(vars(self), dir(self))

    @debug_monitor
    def toggle_weight(self, selection: str, base_weight=1.0, index_num=0) -> None:
        """Determine entry edge, determine index, then adjust weight"""
        entry = [reg_entry for reg_entry in self.intent_graph.edges(data=True) if selection in reg_entry[2].get("entry").model]
        atlas = self.intent_graph[entry[0][0]][entry[0][1]]
        for num in atlas:
            if selection in atlas[num].get("entry").model:
                index_num = num
                base_weight = atlas[index_num].get("weight")
        if int(base_weight) == 0:
            self.intent_graph[entry[0][0]][entry[0][1]][index_num]["weight"] = base_weight + 0.1
        else:
            self.intent_graph[entry[0][0]][entry[0][1]][index_num]["weight"] = base_weight - 0.1

            # [reg_id[2] for reg_id in self.intent_graph.edges(data=True) if selection in reg_id[2].get("entry").model]
        # right = left - weight_value
        # if weight_value
        #     pass

        # left = int(weight_value)
        # right = left - weight_value

    # # def check_weights(self, entry: str) -> None:

    # registry_data = [reg_id[2].get('entry').model for reg_id in self.intent_graph.edges(data=True).model if 'ibm' in reg_id[2].get('entry').model]

    # def
    # add weight
    # check weight

    # async def walk_intent(self, send: bool = False, composer: Callable = None, processor: Callable = None) -> None:
    #     """Provided the coordinates in the intent processor, follow the list of in and out methods"""
    #     await self.confirm_available_graph()
    #     await self.confirm_coordinates_path()
    #     coordinates = self.coordinates_path
    #     if not coordinates:
    #         coordinates = ["text", "text"]
    #     hop_length = len(coordinates) - 1
    #     for i in range(hop_length):
    #         if i + 1 < hop_length:
    #             await self.confirm_coordinates_path()
    #             await self.confirm_model_waypoints()
    #             if send:
    #                 await processor(last_hop=False)
    #                 composer(mode_in=coordinates[i + 1], mode_out=coordinates[i + 2])
    #             else:
    #                 old_model_names = self.model_names if self.model_names else []
    #                 composer(mode_in=coordinates[i + 1], mode_out=coordinates[i + 2], io_only=True)
    #                 self.model_names.extend(old_model_names)

    #         elif send:
    #             await self.confirm_coordinates_path()
    #             await self.confirm_model_waypoints()
    #             processor()
