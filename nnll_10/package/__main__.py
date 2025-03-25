#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=missing-module-docstring, disable=missing-class-docstring

from textual import work, events
from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive

from nnll_01 import info_message as nfo
from nnll_10.package.main_screen import MainScreen  # pylint: disable=import-error
from viztracer import VizTracer
from datetime import datetime
import os
# from theme import fluoresce_theme


class Combo(App):
    """Machine Accelerated Intelligent Network"""

    SCREENS = {"main_on": MainScreen}
    CSS_PATH = "combo.tcss"
    BINDINGS = [Binding("escape", "safe_exit", "◼︎ / ⏏︎")]  # Cancel response

    file_name = f".nnll{datetime.now().strftime('%Y%m%d')}_trace.json"
    assembled_path = os.path.join("log", file_name)
    tracer = VizTracer()
    safety: reactive[int] = reactive(2)

    def on_mount(self) -> None:
        """Draw screen"""
        self.push_screen("main_on")
        self.scroll_sensitivity_y = 1
        self.supports_smooth_scrolling = True
        self.theme = "flexoki"
        self.tracer.start()

    @work(exit_on_error=True)
    async def _on_key(self, event: events.Key):
        """Window for triggering key bindings"""
        if event.key not in ["escape", "ctrl+left_square_brace"]:
            self.safety += 1
        else:
            self.safety -= 1
            if self.safety < 0:
                event.prevent_default()
                self.tracer.stop()
                self.tracer.save(output_file=self.assembled_path)  # also takes output_file as an optional argument
                await self.app.action_quit()


if __name__ == "__main__":
    app = Combo(ansi_color=False)
    nfo("Launching...")
    app.run()
