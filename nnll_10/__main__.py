from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, TextArea, RichLog
from textual.reactive import reactive
from textual.containers import Container

from nnll_11.__main__ import chat


TEXT = """\
Prompt
"""


class SideMargins(Static):
    CSS_PATH = "side_margins.tcss"


class MidPad(Static):
    DEFAULT_CSS = """
    Mid {
    width: 100%;
    }
    """


class MainContainer(Container):
    CSS_PATH = "combo.tcss"


class Combo(App):
    """A Textual app."""

    CSS_PATH = "combo.tcss"
    BINDINGS = [("`", "send_message", "Send Message")]

    scribe: reactive[str] = reactive("", recompose=True)
    model: str = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # yield Header()
        yield Footer()
        yield SideMargins(id="SidebarLeft")
        with MainContainer():
            yield TextArea(TEXT, id="prompts", language="python", theme="vscode_dark", max_checkpoints=100)
            yield MidPad(id="Mid")
            yield RichLog(markup=True)
        yield SideMargins(id="SidebarRight")

    def watch_scribe(self) -> None:
        self.query_one(RichLog).write(f"{self.scribe}")

    def on_ready(self) -> None:
        # self.scroll_sensitivity_y = 0.01
        text_log = self.query_one(RichLog)

        text_log.write(f"[bold magenta] {self.model}:")
        text_log.auto_scroll = True

    async def action_send_message(self) -> None:
        content = self.query_one(TextArea)
        model = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"
        message = content.text
        if message.lower() == "quit" or message.lower() == "exit":
            self.app.exit()
        response = []
        async for chunk in chat(model, message):
            response.append(chunk)
            self.query_one(RichLog).write(response)


if __name__ == "__main__":
    app = Combo()
    app.run()
