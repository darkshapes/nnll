#  # # <!-- // /*  SPDX-License-Identifier: blessing */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual import work
from textual.reactive import reactive
from textual.widgets import DataTable
from litellm.utils import token_counter


class DisplayBar(DataTable):
    token_prefix = "ollama"
    duration: reactive[float] = reactive(0.0, recompose=True)

    def on_mount(self):
        self.show_header = False
        self.show_row_labels = False
        self.cursor_type = None

    @work(exclusive=True)
    async def calculate_tokens(self, model, message, unit_labels):
        """Live display of tokens and characters"""
        model_name = self.token_prefix + model
        token_count = token_counter(model=model_name, messages=[{"content": message}]) - 6
        character_count = len(message)
        self.update_cell_at((0, 0), f"     {character_count}{unit_labels[0]}")
        self.update_cell_at((0, 1), f"{token_count}{unit_labels[1]}")
        self.update_cell_at((0, 2), f"{self.duration}{unit_labels[2]}")

    @work(exclusive=True)
    async def calculate_audio(self, duration, unit_labels):
        """Live display of sound recording length"""
        self.duration = duration
        self.update_cell_at((0, 2), f"{self.duration}{unit_labels[2]}", update_width=True)
