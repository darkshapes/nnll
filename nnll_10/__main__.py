# pylint: disable=missing-module-docstring, disable=missing-class-docstring
from textual import work, events
from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive
from main_screen import WideScreen, TallScreen


class Combo(App):
    """Machine Accelerated Intelligent N"""

    SCREENS = {"widescreen": WideScreen, "tallscreen": TallScreen}
    BINDINGS = [
        Binding("escape", "exit_now()", "(x2) Exit App", priority=True),  # Quit out
    ]
    CSS_PATH = "combo.tcss"

    safety: reactive[int] = reactive(2)

    def on_mount(self) -> None:
        """Draw screen"""
        self.push_screen("tallscreen")

    @work(exit_on_error=False)
    async def on_resize(self, event: events.Resize):
        width = event.container_size.width
        height = event.container_size.height
        if width / 2 >= height:
            self.push_screen("widescreen")
        elif width / 2 < height:
            self.push_screen("tallscreen")

    @work(exit_on_error=False)
    async def _on_key(self, event: events.Key):
        """Window for triggering key bindings"""
        if event.key != "escape":
            self.safety = 3
        else:
            event.prevent_default()
            self.safety -= 1
            if self.safety <= 0:
                event.prevent_default()
                self.exit_now()

    @work(exit_on_error=True)
    async def action_switch_screen(self, screen):
        return await super().action_switch_screen(screen)

    @work(exit_on_error=True)
    async def exit_now(self) -> None:
        """Immediately exit the app"""
        self.app.exit()
        self.app.action_quit()


if __name__ == "__main__":
    app = Combo()
    app.run()
