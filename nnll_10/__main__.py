import os
from textual import work, events, on
from textual.color import Color
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, TextArea, Log, DataTable, Link  # Sparkline, Link,
from textual.reactive import reactive, var
from textual.containers import Container
from textual.binding import Binding
import tiktoken
from nnll_11.__main__ import chat_machine

# from itertools import count
# from numpy import mean

TEXT = """\
Prompt
"""


class SidebarRight(Container):
    pass  # def watch_total_tokens(self):
    #     self.query_one(Link).text = str(self.text)


class SidebarLeft(Container):
    pass


class Combo(App):
    """A Textual app."""

    model = "ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M"
    count = var(0)
    rows = [(0, 0), (" ch  ", " tk  ")]
    status: reactive[str] = reactive(f"Ready {model}")  # mediumvioletred
    EVENT_KEYS = [",", ".", "space"]
    CSS_PATH = "combo.tcss"
    BINDINGS = [Binding("grave_accent", "post_message", "Send Message", priority=True), ("escape", "cancel_worker", "Cancel Processing")]

    def compose(self) -> ComposeResult:
        yield Footer()  # menu
        yield Static(id="SidebarLeft")
        with Container(id="FullScreen"):
            yield TextArea(TEXT, language="python", theme="vscode_dark", max_checkpoints=100)
            yield DataTable(id="DisplayBar")
            # yield Sparkline(self.tokens, summary_function=max)
            with Container(id="SubContainer"):
                yield Log(highlight=True)
                yield Link(self.status, url="", id="SystemStatus")
        yield Static(id="SidebarRight")

    @work(exclusive=True, exit_on_error=False)
    async def _on_key(self, event: events.Key):
        text_field = self.query_one(TextArea)
        if event.key == "grave_accent":
            event.prevent_default()
            text_field.insert("`")
            text_field.move_cursor_relative(columns=-1)
            self.transmit_message(text_field)

    def on_ready(self) -> None:
        # self.scroll_sensitivity_y = 0.01
        text_log = self.query_one(Log)
        # text_log.write()
        text_log.auto_scroll = True
        text_log.scrollbar_background = "surface"
        text_log.wrap = True

        text_box = self.query_one(TextArea)
        text_box.cursor_blink = False
        link = self.query_one(Link)
        link.styles.text_style = "dim"

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.styles.background = Color(0, 0, 0)
        table.styles.back = Color(0, 0, 0)
        table.styles.color = "#333333"
        table.show_row_labels = False
        table.show_header = False
        table.cursor_type = None

    @work(exclusive=True, exit_on_error=False)
    async def action_post_message(self) -> None:
        content = self.query_one(TextArea)
        self.transmit_message(content)

    @work(exclusive=True, exit_on_error=False)
    async def transmit_message(self, content):
        text_log = self.query_one(Log)

        message = content.text
        if message.lower() == "quit" or message.lower() == "exit":
            self.app.exit()
        link = self.query_one(Link)
        link.styles.color = "green"
        self.status = f"Running {self.model}"
        async for chunk in chat_machine(self.model, message):
            text_log.write(chunk)
            text_log.refresh
        link.styles.color = "magenta"

    @on(TextArea.Changed)
    @work(exclusive=True, exit_on_error=False)
    async def calculate_tokens(self):  # field: TextArea.Changed):
        content = self.query_one(TextArea)
        self.count += 1
        encoding = tiktoken.get_encoding("cl100k_base")
        message = content.text
        token_count = len(encoding.encode(message))
        character_count = len(content.text)
        table = self.query_one(DataTable)
        table.update_cell_at((0, 0), f"{character_count}ch")
        table.update_cell_at((0, 1), f"{token_count}tk")

        # totaler.text = self.text

    # async def on_key(self, event: events.Key):
    #     content = self.query_one(TextArea)
    #     event.prevent_default()
    #     content.insert(event.character)
    #     content.move_cursor_relative(columns=-1)
    #     text_log = self.query_one(RichLog)
    #     text_log.write(event)
    # content = self.query_one(TextArea)
    # # event.prevent_default()
    # text_log = self.query_one(RichLog)
    # text_log.write(event)
    # # if event.character.is_printable:
    # #     self.calculate_tokens(event)

    # text: reactive[list[tuple]] = reactive([()])

    # self.query_one(RichLog).write(diff)
    # tokens: reactive[list] = reactive([0])

    #     self.text = str(token_count)
    #     diff = token_count - self.tokens[-1]
    #     if diff <= 0:
    #         diff = 0
    #     self.tokens.append(diff)
    # token_counter = self.query_one(Sparkline)
    # token_counter.data = [self.tokens[count] for count in range(len(self.tokens) - self.count, len(self.tokens))]

    # # totaler = self.query_one(Link)
    # offset = self.count * 120
    # self.tokens.e
    # offset = self.count
    # for x in self.tokens:
    #     data_test = range(0 + offset, len(self.tokens), x)
    # stream = [x for x in self.tokens[]]
    # self.query_one(Sparkline).data = [self.tokens]
    # if event.key in self.EVENT_KEYS:

    # event.prevent_default()
    #   if event.key == "space":
    #         content.insert(" ")
    #     else:
    #         content.insert(next(iter(x for x in self.EVENT_KEYS if event.key == x), ""))


app = Combo()
if __name__ == "__main__":
    app.run()
