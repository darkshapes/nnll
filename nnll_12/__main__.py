#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint:disable=not-an-iterable, wrong-import-position
import sys
import os

sys.path.append(os.getcwd())

import networkx as nx
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Label, ListItem, ListView, RichLog, Static, TextArea

from nnll_01 import debug_message as dbug, debug_monitor
from nnll_05 import pull_path_entries
from nnll_10.package.voice_panel import VoicePanel
from nnll_14 import build_conversion_graph, label_edge_attrib_for, trace_objective

from nnll_15.constants import GenTypeC, GenTypeCText, LibType


class ButtonsApp(App[str]):
    nx_graph: nx.Graph = None
    hover_name: reactive[str] = reactive("")
    start_id: reactive[str] = reactive(None)
    end_id: reactive[str] = reactive(None)
    gen_type: reactive[str] = reactive(None)
    CSS_PATH = "button.tcss"

    def compose(self) -> ComposeResult:
        yield Horizontal(
            VerticalScroll(
                Button("1. Build", id="build", variant="primary"),
                Button("2. Add Attributes", id="attrib", variant="default"),
                Static("3. Source", classes="header"),
                ListView(id="start_points"),
                Static("4. Target", classes="header"),
                ListView(id="end_points"),
                Static("5. Type", classes="header"),
                ListView(id="convert_type"),
            ),
            Vertical(
                TextArea("", id="prompt_pane"),
                VoicePanel(id="speech_pane"),
                TextArea("", id="response_pane", read_only=True),
            ),
            VerticalGroup(
                RichLog(id="results_panel", highlight=True, markup=True, wrap=True),
            ),
        )

    def on_ready(self):
        self.query_one("#results_panel").write("Ready.")

    @debug_monitor
    def on_enter(self, event: events.Enter) -> None:
        self.hover_name = f"{event.node.id}"

    @debug_monitor
    def on_leave(self, event: events.Leave) -> None:
        self.hover_name = f"{event.node.id}"

    @debug_monitor
    def on_mouse_down(self, event: events.MouseEvent) -> None:
        results_panel = self.query_one("#results_panel")
        if self.hover_name == "build":
            # event.stop()
            self.nx_graph = build_conversion_graph()
            results_panel.write(f"Created {self.nx_graph}")
            self.query_one("#attrib").variant = "default"
            self.query_one("#build").variant = "success"
        if self.hover_name == "attrib":
            if not self.nx_graph:
                results_panel.write("Missing step: Build graph")
                self.query_one("#attrib").variant = "warning"
            else:
                self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.OLLAMA)
                self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.HUB)
                available_conversions = {edge[1] for edge in self.nx_graph.edges}
                self.query_one("#start_points").extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                available_conversions = {edge[0] for edge in self.nx_graph.edges}
                self.query_one("#end_points").extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                self.query_one("#attrib").variant = "success"
                results_panel.write(f"Attributes added to {self.nx_graph}")
        if self.hover_name in self.query_children("#start_points"):
            dbug(f"\n{self.query_children('#start_points')}")
        if self.hover_name in self.query_children("#end_points"):
            dbug(f"\n{self.query_children('#end_points')}")
        if self.hover_name in self.query_children("#convert_type"):
            dbug(f"\n{self.query_children('convert_type')}")

    @debug_monitor
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        list_id = event.list_view.id
        results_panel = self.query_one("#results_panel")
        start_type = None
        end_type = None
        if list_id == "start_points":
            end_points = self.query_one("#end_points")
            self.end_id = end_points.index
            dbug(f"end point: {self.end_id}  window : {event.list_view}")
            if self.end_id is not None:
                start_type = event.list_view.highlighted_child.id
                end_type = end_points.highlighted_child.id

        elif list_id == "end_points":
            start_points = self.query_one("#start_points")
            start_id = start_points.index if not None else start_points[0]
            dbug(f"start point: {start_id} window : {event.list_view}")
            if start_id is not None:
                start_type = start_points.highlighted_child.id
                end_type = event.list_view.highlighted_child.id
        elif list_id == "convert_type":
            self.gen_type = f"{event.list_view.highlighted_child.children[0].id}"
            results_panel.write(self.gen_type)

        if start_type is not None and end_type is not None:
            traced_path = trace_objective(nx_graph=self.nx_graph, source=start_type, target=end_type)
            results_panel.write(traced_path)
            registry_entries = pull_path_entries(self.nx_graph, traced_path)
            results_panel.write([x["entry"].model for x in list(registry_entries)])
            convert_type = self.query_one("#convert_type")
            all_fields = GenTypeCText.model_fields | GenTypeC.model_fields
            convert_type.remove_children(ListItem)
            if "text" in start_type or "text" in end_type:
                for param in all_fields:
                    convert_type.append(ListItem(Label(f"{param}", id=f"{param}")))
            else:
                if self.gen_type not in GenTypeC.model_fields:
                    self.gen_type = None
                    results_panel.write(self.gen_type)
                for param in GenTypeC.model_fields:
                    convert_type.append(ListItem(Label(f"{param}", id=f"{param}")))


if __name__ == "__main__":
    app = ButtonsApp()
    app.run()
