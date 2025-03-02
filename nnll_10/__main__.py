import csv
import io
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, TextArea, RichLog
from textual.content import Content
from textual import events, on, containers
from textual.reactive import reactive
from rich.syntax import Syntax
from rich.table import Table

from nnll_10.testrun import chat


TEXT = """\
Write here
"""


CSV = """lane,swimmer,country,time
4,Joseph Schooling,Singapore,50.39
2,Michael Phelps,United States,51.14
5,Chad le Clos,South Africa,51.14
6,László Cseh,Hungary,51.14
3,Li Zhuhao,China,51.26
8,Mehdy Metella,France,51.58
7,Tom Shields,United States,51.73
1,Aleksandr Sadovnikov,Russia,51.84"""


CODE = '''\
def loop_first_last(values: Iterable[T]) -> Iterable[tuple[bool, bool, T]]:
    """Iterate and generate a tuple with a flag for first and last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    first = True
    for value in iter_values:
        yield first, False, previous_value
        first = False
        previous_value = value
    yield first, True, previous_value\
 CSS = """
    Screen {
        content-align: center middle;
        layout: grid;
    }
    TextArea, RichLog {
        width: 100%;
        height: 46fr;
    }
    TextArea {
        margin-top: 0;
    }
    RichLog {
        display: block;
        hatch: left #000000;
        margin-bottom: 0;

    }

'''

class SideMargins(Static):
    CSS_PATH = "side_margins.tcss"

class MidPad(Static):
    DEFAULT_CSS = """
    Mid {
    }
    """

class MainContainer(Static):
    DEFAULT_CSS = """
    Contain {
        display: block;
        width: 100%;
        height: 100%;
        padding: 2;
    }
    """

class Combo(App):
    """A Textual app."""
    CSS_PATH = "combo.tcss"
    BINDINGS = [("`","send_message","Send Message")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # yield Header()
        yield Footer()
        yield SideMargins(id="SidebarLeft")
        with MainContainer():
            yield TextArea(TEXT, id="prompts", language="python", theme="vscode_dark", max_checkpoints=100)
            yield MidPad(id="Mid")
            yield RichLog(id="responses",highlight=True, markup=True)
        yield SideMargins(id="SidebarRight")

    def on_ready(self) -> None:
        self.scroll_sensitivity_y = 0.01
        text_log = self.query_one(RichLog)

        text_log.write(Syntax(CODE, "python", indent_guides=True))

        rows = iter(csv.reader(io.StringIO(CSV)))
        table = Table(*next(rows))
        for row in rows:
            table.add_row(*row)

        text_log.write(table)
        text_log.write("[bold magenta]Write text or any Rich renderable!")
        text_log.auto_scroll = True

    def action_send_message(self) -> None:
        content = self.query_one(TextArea)
        text_log = self.query_one(RichLog)
        text_log.write = chat(content.text)



if __name__ == "__main__":
    app = Combo()
    app.run()

# import sys
# import asyncio

# import dspy
# lm = dspy.LM('ollama_chat/hf.co/xwen-team/Xwen-72B-Chat-GGUF:Q4_K_M', api_base='http://localhost:11434', api_key='')
# dspy.configure(lm=lm)

# lm("Hello, just testing")
# lm(messages=[{"role": "user", "content": "Say this is a test!"}])
