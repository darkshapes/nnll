#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import os
import networkx as nx
from textual import on, events
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Select
from textual.widgets._select import SelectCurrent, SelectOverlay

from nnll_01 import debug_message as dbug  # , debug_monitor


class Selectah(Select):
    models: reactive[list[tuple[str, str]]] = reactive([("", "")])
    graph: nx.Graph = None
    mode_in: str = "text"
    mode_out: str = "text"

    def on_mount(self) -> None:
        # self.options = self.graph.models
        self.graph = self.query_ancestor(Screen).int_proc
        # self.prompt = os.path.basename(next(iter(self.graph.models))[0])

    @on(events.Focus)
    async def on_focus(self) -> None:
        """Expand panel immediately when clicked in terminal"""
        if SelectOverlay.has_focus:
            self.set_options(self.graph.models)

    @on(Select.Changed)
    def on_changed(self) -> None:  # event: Select.Changed) -> None:
        """Rearrange models"""
        try:
            assert self.query_one(SelectCurrent).has_value
        except AssertionError as error_log:
            dbug(error_log)
        else:
            self.graph.edit_weight(selection=self.value, mode_in=self.mode_in, mode_out=self.mode_out)
            self.set_options(self.graph.models)
            self.prompt = next(iter(self.graph.models))[0]


# @on(SelectOverlay.blur)
# def on_select_overlay_blur(self, event: events.Blur) -> None:
#     from rich.text import Text

#     nfo(event.control.id, "blah blah blah")
#     # if event.control == SelectOverlay:
#     self.ui["sl"].prompt = next(iter(self.int_proc.models))[0]
#     label = self.ui["sl"].query_one("#label", Static)
#     try:
#         assert label.renderable == next(iter(self.int_proc.models))[0]
#     except AssertionError as error_log:
#         dbug(error_log)
#         self.ui["sl"].prompt = next(iter(self.int_proc.models))[0]

# @on(OptionList.OptionSelected)
# def on_select_overlay_option_selected(self, event: OptionList.OptionSelected) -> None:
#     """Textual API event, refresh pathing, Switch checkpoint assignments"""
#     nfo(" test write")
#     overlay = self.ui["sl"].query_one(SelectOverlay)
#     mode_in = self.ui["it"].get_cell_at((self.ui["it"].current_row, 1))
#     mode_out = self.ui["ot"].get_cell_at((self.ui["ot"].current_row, 1))
#     if self.ui["sl"].selection == Select.BLANK or self.ui["sl"].selection is None:
#         selection = next(iter(self.int_proc.models))[1]
#     else:
#         selection = self.ui["sl"].selection
#     self.int_proc.edit_weight(selection=selection, mode_in=mode_in, mode_out=mode_out)
#     self.ready_tx()
#     self.walk_intent()
#     overlay.recompose()
#     # self.ui["sl"].set_options(self.int_proc.models)

# self.ready_tx()
# if self.int_proc.has_graph() and self.int_proc.has_path():
#     self.walk_intent()
# if self.int_proc.has_ckpt():

# self.ui["sl"].set_options(options=self.int_proc.models)
# nfo(self.int_proc.has_ckpt())
# self.ui["sl"].expanded = False

# @work(exclusive=True)
