#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Orientations"""

from textual import work, events
from textual.app import ComposeResult
from textual.containers import Horizontal, Container
from textual.screen import Screen
from textual.widgets import Footer, Static

from .fold import Fold  # pylint: disable=import-error


# class WideScreen(Screen):
class MainScreen(Screen):
    """Orienting display Horizontal"""

    # Laptop layout."""

    def compose(self) -> ComposeResult:
        yield Footer(id="footer")
        with Horizontal(id="app-grid"):
            yield ResponsiveLeftTop(id="left-frame")
            yield Fold(id="centre-frame")
            yield ResponsiveRightBottom(id="right-frame")

    def on_mount(self):
        self.query_one("#app-grid").set_classes("app-grid-horizontal")

    @work(exit_on_error=False)
    async def on_resize(self, event=events.Resize):
        """Fit shape to screen"""
        display = self.query_one("#app-grid")
        width = event.container_size.width
        height = event.container_size.height
        if width / 2 >= height:  # Screen is wide
            display.set_classes("app-grid-horizontal")
        elif width / 2 < height:  # Screen is tall
            display.set_classes("app-grid-vertical")


# class TallScreen(Screen):
#     """Orienting display Vertical
#     Phone layout."""

#     def compose(self) -> ComposeResult:
#         yield Footer()
#         with Vertical(id="app-grid-vertical"):
#             yield ResponsiveLeftTop(id="top-frame")
#             yield Fold(id="centre-frame")
#             yield ResponsiveRightBottom(id="bottom-frame")


class ResponsiveLeftTop(Container):
    """Sidebar Left/Top"""

    def compose(self) -> ComposeResult:
        yield Static()


class ResponsiveRightBottom(Container):
    """Sidebar Right/Bottom"""

    def compose(self) -> ComposeResult:
        yield Static()
        yield Static()
