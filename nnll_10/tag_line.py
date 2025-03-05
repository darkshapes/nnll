#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from textual import work
from textual.reactive import reactive
from textual.widgets import Link


class TagLine(Link):
    """A floating tag indicator"""

    model = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"  # example default (drag/drop model to load from server?)
    model_short = os.path.basename(model)  # shorter for display
    status: reactive[str] = reactive(f"{model_short}")  # mediumvioletred # Chromatic status indication

    @work(exit_on_error=False)
    async def update_status(self) -> None:
        """Provide visual status update of processing"""
        self.styles.color = "magenta" if self.styles.color == "bold green" else "bold green"

    @work(exit_on_error=False)
    async def update_model(self, model: str) -> None:
        """Provide visual status update of processing"""
        self.model_short = model
