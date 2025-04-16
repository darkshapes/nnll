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

from nnll_01 import debug_message as dbug, info_message as nfo
from nnll_01 import debug_monitor
from nnll_10 import IntentProcessor
from nnll_10.package.response_panel import ResponsePanel
from nnll_10.package.token_counters import tk_count
from nnll_13 import VoicePanel
from nnll_15.constants import GenTypeC, GenTypeCText
from nnll_19 import MessagePanel

# from nnll_20 import ResponsePanel


class ButtonsApp(App[str]):
    intent_processor: nx.Graph = None
    hover_name: reactive[str] = reactive("")
    start_id: reactive[str] = reactive(None)
    end_id: reactive[str] = reactive(None)
    gen_type: reactive[str] = reactive(None)
    registry_entries: reactive[list[str]] = reactive([])
    tokenizer: reactive[str] = reactive("")
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
            self.intent_processor = IntentProcessor()
            self.intent_processor.calculate_intent_graph()
            results_panel.write(f"Created {self.intent_processor}")
            start_points = self.query_one("#start_points")
            end_points = self.query_one("#end_points")

            if not self.intent_processor:
                results_panel.write("Missing step: Build graph")
                build_button.variant = "warning"
            elif not start_points.children and not end_points.children:
                available_conversions = {edge[1] for edge in self.intent_processor.intent_graph.edges}
                start_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                available_conversions = {edge[0] for edge in self.intent_processor.intent_graph.edges}
                end_points.extend([ListItem(Label(f"{edge}"), id=f"{edge}") for edge in available_conversions])
                build_button.variant = "success"
                results_panel.write(f"Attributes added to {self.intent_processor}")
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
            self.intent_processor.derive_coordinates_path(mode_in=start_type, mode_out=end_type)
            results_panel.write(self.intent_processor.coordinates_path)
            self.intent_processor.define_model_waypoints()
            self.tokenizer = next(iter(self.intent_processor.model_names))
            nfo(self.intent_processor.intent_graph.nodes(data=True))
            nfo([*self.intent_processor.intent_graph.edges(data=True)])

            self.query_one("#response_panel").insert(f"{str(self.tokenizer)}\n")
            results_panel.write([x["entry"].model for x in list(self.intent_processor.registry_entries)])
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
        response_panel = self.query_one("#response_panel")
        message = self.query_one("#message_panel").text
        token_count = await tk_count(self.tokenizer, message)
        response_panel.clear()
        response_panel.insert(f"{str(self.tokenizer)}\n{str(token_count)}")

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
