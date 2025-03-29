#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual.reactive import reactive

# from nnll_01 import debug_monitor
from nnll_10.package.carousel import Carousel


class InputTag(Carousel):
    """Input Types"""

    available_inputs: dict = {"▲ Prompt ▼": "message_panel", "▲ Speech ▼": "voice_panel"}
    current_input: reactive[str] = reactive("")

    def on_mount(self):
        self.current_input = self.available_inputs.get(next(iter(self.available_inputs)))
        self.add_columns(("0", "1"))
        self.add_rows([row.strip()] for row in self.available_inputs)
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
