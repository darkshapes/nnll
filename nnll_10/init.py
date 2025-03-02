import csv
import io
from textual._text_area_theme import TextAreaTheme
from textual.app import App, ComposeResult, Screen
from textual.widgets import Footer, Static, TextArea, Placeholder
from textual import events
from textual.widgets import RichLog
from rich.syntax import Syntax
from rich.table import Table



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

    BINDINGS = ["shift+enter","",""]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # yield Header()
        yield Footer()
        yield SideMargins(id="SidebarLeft")
        with MainContainer():
            yield TextArea(TEXT, language="python", theme="vscode_dark", max_checkpoints=100)
            yield MidPad(id="Mid")
            yield RichLog(highlight=True, markup=True)
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


if __name__ == "__main__":
    app = Combo()
    app.run()
