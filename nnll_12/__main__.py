#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, Vertical, VerticalGroup
from textual.widgets import Button, Static, RichLog, ListView, ListItem, Label, TextArea
import networkx as nx

from nnll_05 import pull_path_entries
from nnll_10.package.voice_panel import VoicePanel
from nnll_14 import build_conversion_graph, label_edge_attrib_for, trace_objective
from nnll_15.constants import LibType


class ButtonsApp(App[str]):
    nx_graph: nx.Graph = None
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
            ),
            VerticalGroup(
                RichLog(id="results_panel", highlight=True, markup=True, wrap=True),
            ),
            Vertical(
                TextArea("", id="prompt_pane"),
                VoicePanel(id="speech_pane"),
                TextArea("", id="response_pane", read_only=True),
            ),
        )

    def on_ready(self):
        self.query_one("#results_panel").write("Ready.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        write_args = ""
        if event.button.id == "build":
            self.nx_graph = build_conversion_graph()
            write_args = f"Created {self.nx_graph}"
            event.button.variant = "success"
        if event.button.id == "attrib":
            if not self.nx_graph:
                write_args = "Missing step: Build graph"
            else:
                self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.OLLAMA)
                self.nx_graph = label_edge_attrib_for(self.nx_graph, LibType.HUB)
                event.button.variant = "success"
                write_args = f"Attributes added to {self.nx_graph}"
                available_conversions = {edge[1] for edge in self.nx_graph.edges}
                start_points = self.query_one("#start_points")
                start_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                self.query_one("#end_points")
                available_conversions = {edge[0] for edge in self.nx_graph.edges}
                end_points = self.query_one("#end_points")
                end_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])

        self.query_one("#results_panel").write(write_args)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self.query_one("#results_panel").write(event.list_view.id)
        results_panel = self.query_one("#results_panel")
        if event.list_view.id == "start_points":
            end_points = self.query_one("#end_points")
            end_id = end_points.index
            results_panel.write(f"end point: {end_id}  window : {event.list_view}")
            if end_id is not None:
                results_panel.write(f"start : {event.list_view.highlighted_child.id} end: {end_points.highlighted_child.id} ")
                traced_path = trace_objective(nx_graph=self.nx_graph, source=event.list_view.highlighted_child.id, target=end_points.highlighted_child.id)
                results_panel.write(traced_path)
                registry_entries = pull_path_entries(self.nx_graph, traced_path)
                results_panel.write(list(registry_entries))

        elif event.list_view.id == "end_points":
            start_points = self.query_one("#start_points")
            start_id = start_points.index if not None else start_points[0]
            results_panel.write(f"start point: {start_id} window : {event.list_view}")
            if start_id is not None:
                results_panel.write(f"start : {start_points.highlighted_child.id}  end :{event.list_view.highlighted_child.id}")
                traced_path = trace_objective(nx_graph=self.nx_graph, source=start_points.highlighted_child.id, target=event.list_view.highlighted_child.id)
                results_panel.write(traced_path)
                registry_entries = pull_path_entries(self.nx_graph, traced_path)
                results_panel.write(list(registry_entries))


if __name__ == "__main__":
    app = ButtonsApp()
    app.run()
