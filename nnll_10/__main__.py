# pylint: disable=missing-module-docstring, disable=missing-class-docstring

from textual.app import App
from .full_screen import FullScreen


class Combo(App):
    """A Textual app."""

    SCREENS = {"fullscreen": FullScreen}
    CSS_PATH = "combo.tcss"

    def on_mount(self) -> None:
        """Draw screen"""
        self.push_screen("fullscreen")


if __name__ == "__main__":
    app = Combo()
    app.run()
