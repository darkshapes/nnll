### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


"""建立控制檯日誌 logging"""

import logging as py_logging
import sys
from functools import wraps
from logging import Formatter, StreamHandler
from typing import Callable

from rich import console, style, theme
from rich import logging as rich_logging

from nnll_01 import LOG_LEVEL

msg_init = None  # pylint: disable=invalid-name

NOUVEAU = theme.Theme(
    {
        "logging.level.notset": style.Style(dim=True),  # level ids
        "logging.level.debug": style.Style(color="magenta3"),
        "logging.level.info": style.Style(color="blue_violet"),
        "logging.level.warning": style.Style(color="gold3"),
        "logging.level.error": style.Style(color="dark_orange3", bold=True),
        "logging.level.critical": style.Style(color="deep_pink4", bold=True, reverse=True),
        "logging.keyword": style.Style(bold=True, color="cyan", dim=True),
        "log.path": style.Style(dim=True, color="royal_blue1"),  # line number
        "repr.str": style.Style(color="sky_blue3", dim=True),
        "json.str": style.Style(color="gray53", italic=False, bold=False),
        "log.message": style.Style(color="steel_blue1"),  # variable name, normal strings
        "repr.tag_start": style.Style(color="white"),  # class name tag
        "repr.tag_end": style.Style(color="white"),  # class name tag
        "repr.tag_contents": style.Style(color="deep_sky_blue4"),  # class readout
        "repr.ellipsis": style.Style(color="purple4"),
        "log.level": style.Style(color="gray37"),
    }
)
console_out = console.Console(stderr=True, theme=NOUVEAU)

log_handler = rich_logging.RichHandler(console=console_out)

if log_handler is None:
    log_handler = StreamHandler(sys.stderr)
    log_handler.propagate = False

formatter = Formatter(
    fmt="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log_handler.setFormatter(formatter)
py_logging.root.setLevel(LOG_LEVEL)
py_logging.root.addHandler(log_handler)


if msg_init is not None:
    logger = py_logging.getLogger(__name__)
    logger.info(msg_init)

log_level = getattr(py_logging, LOG_LEVEL)
logger = py_logging.getLogger(__name__)


def debug_monitor(func):
    """Debug output decorator function
    Data returned from decorated methods/functions is automatically sent to debugger
    """

    def wrapper(*args, **kwargs) -> None:
        """Wrap log"""
        return_data = func(*args, **kwargs)
        if not kwargs:
            logger.debug(
                "%s",
                f"Func {func.__name__} : {type(args)} : {args} : Return : {return_data}",
            )
        else:
            logger.debug(
                "%s",
                f"Func {func.__name__}{type(args)}:{args}:{type(kwargs)}:{kwargs}:R{return_data}",
            )
        return return_data

    return wrapper


def info_monitor(*args):
    """Info log output"""
    logger.info(" ".join(map(str, args)), exc_info=LOG_LEVEL)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception in function", exc_info=(exc_type, exc_value, exc_traceback))


class CaptureStdOut:
    def __init__(self, logger_func: Callable, level=LOG_LEVEL):
        """Capture console output and send automatically to logger
        :param logger_func: The logging function (e.g., logger.info)
        :param level: Logging level to use when writing logs.
        """
        self.logger = logger_func
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger(line)

    def flush(self):
        # This may be needed by some libraries that expect a flush method.
        pass


def capture_io(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save original stdout and stderr
        # Redirect stdout, stderr,custom exception handler to the logger
        # Restore original stdout, stderr, exception hook
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_excepthook = sys.excepthook

        try:
            sys.stdout = CaptureStdOut(info_monitor)
            sys.stderr = CaptureStdOut(info_monitor, level=py_logging.ERROR)

            sys.excepthook = handle_exception

            return func(*args, **kwargs)

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            sys.excepthook = original_excepthook

    return wrapper


def debug_message(*args):
    """Individual Debug messages"""
    logger.debug(args)
