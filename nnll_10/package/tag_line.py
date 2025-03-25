#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from textual.reactive import reactive

# from nnll_01 import debug_monitor
from nnll_10.package.model_register import from_ollama_cache

from nnll_10.package.carousel import Carousel


class TagLine(Carousel):
    available_models: reactive[dict] = reactive({})
    current_model: reactive[str] = reactive("")
    model_prefix = "ollama_chat/"
    token_prefix = "ollama"

    def on_mount(self):
        self.available_models = from_ollama_cache()
        self.current_model = self.available_models.get(next(iter(self.available_models)))
        self.add_columns(("0", "1"))
        self.add_rows([row.strip()] for row in self.available_models)
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
