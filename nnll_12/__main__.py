#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint:disable=not-an-iterable, wrong-import-position, unsupported-membership-test
import os
import sys


sys.path.append(os.getcwd())

import networkx as nx
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Label, ListItem, ListView, RichLog, Static

from nnll_01 import debug_message as dbug
from nnll_01 import debug_monitor
from nnll_05 import pull_path_entries
from nnll_13 import VoicePanel
from nnll_14 import calculate_graph, trace_objective
from nnll_15.constants import GenTypeC, GenTypeCText, LibType
from nnll_19 import MessagePanel
from nnll_20 import ResponsePanel


class ButtonsApp(App[str]):
    nx_graph: nx.Graph = None
    hover_name: reactive[str] = reactive("")
    start_id: reactive[str] = reactive(None)
    end_id: reactive[str] = reactive(None)
    gen_type: reactive[str] = reactive(None)
    registry_entries: reactive[list[str]] = reactive([])
    traced_path: reactive[list[str]] = reactive([])

    CSS_PATH = "button.tcss"

    # BINDINGS = [
    #     Binding("`", "scribe_response", "go", priority=True),  # Send to LLM
    #     Binding("bk", "", "⌨️"),
    #     Binding("alt+bk", "clear_input", "del"),
    #     Binding("escape", "cancel_generation", "◼︎ / ⏏︎"),  # Cancel response
    # ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            VerticalScroll(
                Button("Build", id="build", variant="primary"),
                # Button("2. Add Attributes", id="attrib", variant="default"),
                # Static("3. Source", classes="header"),
                ListView(id="start_points"),
                Static("to", classes="header"),
                ListView(id="end_points"),
                # Static("5. Type", classes="header"),
                ListView(id="convert_type", initial_index=0),
            ),
            Vertical(
                MessagePanel("", id="message_panel", max_checkpoints=100),
                VoicePanel(id="speech_pane"),
                ResponsePanel("\n", id="response_panel", language="markdown"),
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
    async def on_mouse_down(self, event: events.MouseEvent) -> None:
        results_panel = self.query_one("#results_panel")
        build_button = self.query_one("#build")
        if self.hover_name == "build":
            # event.stop()
            self.nx_graph = calculate_graph()
            results_panel.write(f"Created {self.nx_graph}")
            start_points = self.query_one("#start_points")
            end_points = self.query_one("#end_points")
            if not self.nx_graph:
                results_panel.write("Missing step: Build graph")
                build_button.variant = "warning"
            elif not start_points.children and not end_points.children:
                available_conversions = {edge[1] for edge in self.nx_graph.edges}
                start_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                available_conversions = {edge[0] for edge in self.nx_graph.edges}
                end_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                build_button.variant = "success"
                results_panel.write(f"Attributes added to {self.nx_graph}")
            else:
                build_button.variant = "warning"
                results_panel.write("Attributes already added to graph")
        elif "start_points" in self.hover_name:
            results_panel.write(self.hover_name)
        elif self.hover_name in self.query_children("#end_points"):
            dbug(f"\n{self.query_children('#end_points')}")
        elif self.hover_name in self.query_children("#convert_type"):
            dbug(f"\n{self.query_children('convert_type')}")

    @debug_monitor
    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
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
            self.traced_path = trace_objective(nx_graph=self.nx_graph, source=start_type, target=end_type)
            results_panel.write(self.traced_path)
            self.registry_entries = pull_path_entries(self.nx_graph, self.traced_path)
            results_panel.write([x["entry"].model for x in list(self.registry_entries)])
            convert_type = self.query_one("#convert_type")
            all_fields = GenTypeCText.model_fields | GenTypeC.model_fields
            convert_type.remove_children(ListItem)
            self.gen_type = "translate"
            results_panel.write(self.gen_type)
            results_panel.write(event.item.walk_children())
            if "text" in start_type or "text" in end_type:
                for param in sorted(all_fields, reverse=True):
                    convert_type.append(ListItem(Label(f"{param}", id=f"{param}")))
            else:
                for param in sorted(GenTypeC.model_fields, reverse=True):
                    convert_type.append(ListItem(Label(f"{param}", id=f"{param}")))

    @work(exclusive=True)
    async def _on_key(self, event: events.Key):
        """Class method, window for triggering key bindings"""
        if (hasattr(event, "character") and event.character == "`") or event.key == "grave_accent":
            event.prevent_default()
            message = self.query_one("#message_panel").text
            # audio_sample = self.query_one("#voice_panel").audio
            # image_sample = self.query_one("#image_panel").file_name # probably drag and drop this
            # content = {
            #     "text": message if message and len(message) > 0 else None,
            #     "audio": audio_sample if audio_sample and len(audio_sample) > 0 else None,
            #     "image": image_sample if image_sample and len(image_sample) > 0 else None,
            # }
            # self.traced_path("", id="response_pane", read_only=True),
            self.query_one("#tag_line").add_class("active")
            response_panel = self.query_one("#response_panel")
            response_panel.scribe_response(self.traced_path, message)  # , content, target)
            self.query_one("#tag_line").set_classes(["tag_line"])
        elif event.key == "escape":  # self.query_one("#responsive_input").has_focus_within and
            # self.query_one("#response_panel").focus()
            self.cancel_generation()

    @work(exclusive=True)
    async def cancel_generation(self) -> None:
        """Stop the processing of a model"""
        self.query_one("#response_panel").workers.cancel_all()
        self.query_one("#tag_line").set_classes("tag_line")

    @work(exclusive=True)
    async def clear_input(self):
        """Clear the input on the focused panel"""
        if self.query_one("#voice_panel").has_focus:
            self.query_one("#voice_panel").erase_audio()
            # self.pass_audio_to_tokenizer()
        elif self.query_one("#message_panel").has_focus:
            self.query_one("#message_panel").erase_message()


if __name__ == "__main__":
    app = ButtonsApp()
    app.run()
