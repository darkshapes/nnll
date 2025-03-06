#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Orientations"""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.screen import Screen
from textual.widgets import Footer, Static

from fold import Fold  # pylint: disable=import-error


class WideScreen(Screen):
    """Orienting display Horizontal
    Laptop layout."""

    def compose(self) -> ComposeResult:
        yield Footer(id="footer")
        with Horizontal(id="app-grid-horiziontal"):
            yield ResponsiveLeftTop(id="left-frame")
            yield Fold(id="centre-frame")
            yield ResponsiveRightBottom(id="right-frame")


class TallScreen(Screen):
    """Orienting display Vertical
    Phone layout."""

    def compose(self) -> ComposeResult:
        yield Footer()
        with Vertical(id="app-grid-vertical"):
            yield ResponsiveLeftTop(id="top-frame")
            yield Fold(id="centre-frame")
            yield ResponsiveRightBottom(id="bottom-frame")


class ResponsiveLeftTop(Container):
    """Sidebar Left/Top"""

    def compose(self) -> ComposeResult:
        yield Static()


class ResponsiveRightBottom(Container):
    """Sidebar Right/Bottom"""

    def compose(self) -> ComposeResult:
        yield Static()
        yield Static()
