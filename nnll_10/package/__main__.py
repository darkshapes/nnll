#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=missing-module-docstring, disable=missing-class-docstring

from textual import work, events
from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive

from nnll_01 import info_message as nfo
from nnll_10.package.main_screen import MainScreen  # pylint: disable=import-error

# from theme import fluoresce_theme


class Combo(App):
    """Machine Accelerated Intelligent Network"""

    SCREENS = {"main_on": MainScreen}
    CSS_PATH = "combo.tcss"
    BINDINGS = [Binding("escape", "safe_exit", "◼︎ / ⏏︎")]  # Cancel response

    safety: reactive[int] = reactive(1)

    def on_mount(self) -> None:
        """Draw screen"""
        self.push_screen("main_on")
        self.scroll_sensitivity_y = 1
        self.supports_smooth_scrolling = True
        self.theme = "flexoki"

    @work(exit_on_error=True)
    async def _on_key(self, event: events.Key):
        """Window for triggering key bindings"""
        if event.key not in ["escape", "ctrl+left_square_brace"]:
            self.safety += 1
        else:
            self.safety -= 1
            if self.safety < 0:
                event.prevent_default()

                await self.app.action_quit()


if __name__ == "__main__":
    import sys

    trace = False
    if sys.argv[0] == "-t" or sys.argv[0] == "--trace":
        from viztracer import VizTracer

        trace = True
        tracer = VizTracer()
        tracer.start()

    app = Combo(ansi_color=False)

    nfo("Launching...")
    app.run()
    if trace:
        import os
        from datetime import datetime

        os.makedirs("log", exist_ok=True)
        assembled_path = os.path.join("log", f".nnll{datetime.now().strftime('%Y%m%d')}_trace.json")
        tracer.stop()
        tracer.save(output_file=assembled_path)  # also takes output_file as an optional argument
