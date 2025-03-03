from itertools import count
from numpy import mean
from sympy import sequence
from textual import work, events
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, TextArea, RichLog, Sparkline
from textual.reactive import reactive, var
from textual.containers import Container
import tiktoken
from nnll_11.__main__ import chat_machine


TEXT = """\
Prompt
"""


class SubContainer(Container):
    pass


class Combo(App):
    """A Textual app."""

    scribe: reactive[str] = reactive("", recompose=True)
    tokens: reactive[list] = reactive([1, 1])
    model: str = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"
    count = var(0)

    EVENT_KEYS = ["space", ".", ","]
    CSS_PATH = "combo.tcss"
    BINDINGS = [("`", "post_message", "Send Message"), ("escape", "cancel_worker", "Cancel Processing")]
    header = f"""{model}"""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # yield Header()
        yield Footer()
        yield Static(id="SidebarLeft")
        with Container(id="FullScreen"):
            yield TextArea(TEXT, id="prompts", language="python", theme="vscode_dark", max_checkpoints=100)
            yield Sparkline(self.tokens, summary_function=mean)
            yield RichLog(markup=True)
        yield Static(id="SidebarRight")

    def on_ready(self) -> None:
        # self.scroll_sensitivity_y = 0.01
        text_log = self.query_one(RichLog)
        text_log.write(f"[bold magenta] Ready : {self.header}")
        text_log.auto_scroll = True

        def watch_tokens(self) -> None:
            self.query_one(Sparkline).data = [self.tokens]

    async def action_post_message(self) -> None:
        content = self.query_one(TextArea)
        self.transmit_message(content)

    @work(exclusive=True, exit_on_error=False)
    async def transmit_message(self, content):
        message = content.text
        if message.lower() == "quit" or message.lower() == "exit":
            self.app.exit()
        async for chunk in chat_machine(self.model, message):
            self.query_one(RichLog).write(chunk)

    async def on_key(self, event: events.Key):
        if event.key in self.EVENT_KEYS:
            event.prevent_default()
            content = self.query_one(TextArea)
            content.insert(event.key)
            content.move_cursor_relative(columns=-1)
            self.count += 1
            encoding = tiktoken.get_encoding("cl100k_base")
            message = content.text  # get_text_range(position, line_length)
            token_count = [len(encoding.encode(message))]
            self.tokens.extend(token_count)
            self.query_one(RichLog).write(self.tokens)
            token_counter = self.query_one(Sparkline)

            token_counter.data = [count for count in range(len(self.tokens) - self.count, len(self.tokens))]

        # offset = self.count * 120
        # self.tokens.e
        # offset = self.count
        # for x in self.tokens:
        #     data_test = range(0 + offset, len(self.tokens), x)
        # stream = [x for x in self.tokens[]]
        # self.query_one(Sparkline).data = [self.tokens]


if __name__ == "__main__":
    app = Combo()
    app.run()
