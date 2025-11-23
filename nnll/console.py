# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
from sys import stderr as sys_stderr

from logging import DEBUG, CRITICAL, INFO, StreamHandler, getLogger, Formatter  # noqa: F401, pylint:disable=unused-import

from rich.console import Console
from rich.logging import RichHandler


def wipe_printer(*formatted_data: dict) -> None:
    """Print data sp that it replaces itself in the console buffer\n
    :param formatted_data: Output of `pretty_tabled_output()`"""
    from sys import stdout

    stdout.write("\033[F\r" * (len(formatted_data)))  # ANSI escape codes to move the cursor up `len`` lines
    for line_data in formatted_data:
        stdout.write(" " * 175 + "\x1b[1K\r")
        stdout.write(f"{line_data}\n")  # Print the lines

    stdout.flush()  # Empty output buffer to ensure the changes are shown


def pretty_tabled_output(table_title: str, aggregate_data: dict, width: int = 18) -> None:
    """Pretty print data in column format\n
    :param table_title: Header key to use for the table
    :param aggregate_data: A dictionary of values to print
    :param width: The character width of the table
    :return: A formatted bundle of data ready to print"""
    table_contents = aggregate_data
    key_value_length = len(table_contents)
    # width_top = key_value_length * 1.5
    info_format = "{:^{width}}|" * key_value_length
    header_keys = tuple(table_contents)
    horizontal_bar = "  " + "-" * (width - 1) * key_value_length
    horizontal_bar += "----"
    formatted_data = tuple(table_contents.values())
    wipe_printer(table_title, info_format.format(*header_keys, width=width), horizontal_bar, info_format.format(*formatted_data, width=width))


def info_stream():
    """info console logging\n
    :return: INFO level logging object"""
    try:
        console_out = Console(stderr=True)
        log_handler = RichHandler(console=console_out)
        if log_handler is None:
            log_handler = StreamHandler(sys_stderr)
            log_handler.propagate = False
        formatter = Formatter(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_handler.setFormatter(formatter)
        info_log = getLogger(name="nfo")
        info_log.setLevel(INFO)
        info_log.addHandler(log_handler)
        return info_log
    except ImportError:
        pass


INFO_OBJ = info_stream()


def nfo(*args, **kwargs) -> None:  # pylint:disable=unused-argument
    """Info log output"""
    try:
        INFO_OBJ.info("%s", f"{args}")
    except ImportError:
        pass
