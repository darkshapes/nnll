### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""初始化 init, argparse, 建立控制檯日誌 logging"""

# pylint: disable=line-too-long

import argparse
import logging as py_logging
import sys
from importlib import metadata
from logging import Formatter, StreamHandler

from rich import console, style, theme
from rich import logging as rich_logging

msg_init = None  # pylint: disable=invalid-name

# from re import I  # setuptools-scm versioning

if "pytest" not in sys.modules:
    parser = argparse.ArgumentParser(description="Set logging level.")
    group = parser.add_mutually_exclusive_group()

    levels = {"d": "DEBUG", "w": "WARNING", "e": "ERROR", "c": "CRITICAL", "i": "INFO"}
    choices = list(levels.keys()) + list(levels.values()) + [value.upper() for value in levels.values()]
    for short, long in levels.items():
        group.add_argument(f"-{short}", f"--{long.lower()}", f"--{long}", action="store_true", help=f"Set logging level {long}")

    group.add_argument("--log-level", default="i", type=str, choices=choices, help=f"Set the logging level ({choices})")

    args = parser.parse_args()

    # Resolve log_level from args dynamically
    LOG_LEVEL = levels[next(iter([k for k, v in levels.items() if getattr(args, v.lower(), False)]), args.log_level)]
else:
    LOG_LEVEL = "DEBUG"


try:
    __version__ = metadata.version("nnll")
except metadata.PackageNotFoundError as error_log:
    print(f"dataset-tools package is not installed. Did you run `pip install .`? {error_log}")


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


def debug_message(*args):
    """Individual Debug messages"""
    logger.debug(args)


# ### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->


# """初始化 init, argparse, 建立控制檯日誌 logging"""
# # pylint: disable=line-too-long, invalid-name, import-outside-toplevel

# from functools import lru_cache
# from logging import Logger
# from sys import modules as sys_modules
# from typing import Callable

# from rich import style
# from rich.theme import Theme

# NOUVEAU: Theme = Theme(
#     {
#         "logging.level.notset": style.Style(dim=True),  # level ids
#         "logging.level.debug": style.Style(color="magenta3"),
#         "logging.level.info": style.Style(color="blue_violet"),
#         "logging.level.warning": style.Style(color="gold3"),
#         "logging.level.error": style.Style(color="dark_orange3", bold=True),
#         "logging.level.critical": style.Style(color="deep_pink4", bold=True, reverse=True),
#         "logging.keyword": style.Style(bold=True, color="cyan", dim=True),
#         "log.path": style.Style(dim=True, color="royal_blue1"),  # line number
#         "repr.str": style.Style(color="sky_blue3", dim=True),
#         "json.str": style.Style(color="gray53", italic=False, bold=False),
#         "log.message": style.Style(color="steel_blue1"),  # variable name, normal strings
#         "repr.tag_start": style.Style(color="white"),  # class name tag
#         "repr.tag_end": style.Style(color="white"),  # class name tag
#         "repr.tag_contents": style.Style(color="deep_sky_blue4"),  # class readout
#         "repr.ellipsis": style.Style(color="purple4"),
#         "log.level": style.Style(color="gray37"),
#     }
# )


# @lru_cache
# def assign_logging_to(file_name: str = ".nnll", folder_path_named: str = "log") -> Logger:
#     """
#     Configure and launch logger\n
#     :param file_name: The desired filename of the log file
#     :param folder_path_named: The desired path of the folder
#     :return: a `logging` object ready to use for tracking operations
#     """
#     from datetime import datetime
#     from logging import FileHandler, Formatter, getLogger
#     from logging import root as logging_root
#     from os import makedirs
#     from os.path import join

#     from rich.console import Console
#     from rich.logging import RichHandler

#     file_time_str = "%H:%M:%S"
#     file_date_str = datetime.now().strftime("%Y%m%d")
#     file_name += f"{file_date_str}"
#     makedirs(folder_path_named, exist_ok=True)
#     assembled_path = join(folder_path_named, file_name)

#     file_output = FileHandler(assembled_path, "a+", encoding="utf-8")
#     formatter = Formatter(fmt="%(message)s", datefmt=file_time_str)
#     console_out = Console(stderr=True, theme=NOUVEAU)

#     log_handler = RichHandler(level=LOG_LEVEL, rich_tracebacks=True, tracebacks_show_locals=True, console=console_out, show_time=True, log_time_format=file_time_str)
#     file_output.setFormatter(formatter)
#     logging_root.setLevel(LOG_LEVEL)
#     logging_root.addHandler(file_output)
#     logging_root.addHandler(log_handler)
#     logger = getLogger(__name__)

#     return logger


# def debug_monitor(func: Callable = None) -> Callable:
#     """Debug output decorator function
#     Data returned from decorated methods/functions is automatically sent to debugger
#     """

#     def wrapper(*args, **kwargs) -> None:
#         """Wrap log"""
#         if "pytest" not in sys_modules:
#             return_data = func(*args, **kwargs)
#             if not kwargs:
#                 logger_obj.debug(
#                     "%s",
#                     f"Func {func.__name__} : {type(args)} : {args} : Return : {return_data}",
#                 )
#             else:
#                 logger_obj.debug(
#                     "%s",
#                     f"Func {func.__name__}{type(args)}:{args}:{type(kwargs)}:{kwargs}:R{return_data}",
#                 )
#             return return_data

#     return wrapper


# def info_monitor(*args):
#     """Info log output"""
#     if "pytest" not in sys_modules:
#         logger_obj.info("%s", args, exc_info=True)


# def debug_message(*args):
#     """Info log output"""
#     if "pytest" not in sys_modules:
#         logger_obj.debug("%s", args, exc_info=True)


# if __name__ == "__main__":
#     import argparse
#     from logging import DEBUG as _DEBUG

#     if "pytest" not in sys_modules:
#         parser = argparse.ArgumentParser(description="Set logging level.")
#         group = parser.add_mutually_exclusive_group()

#         levels = {"d": "DEBUG", "w": "WARNING", "e": "ERROR", "c": "CRITICAL", "i": "INFO"}
#         choices = list(levels.keys()) + list(levels.values()) + [value.upper() for value in levels.values()]
#         for short, long in levels.items():
#             group.add_argument(f"-{short}", f"--{long.lower()}", f"--{long}", action="store_true", help=f"Set logging level {long}")

#         group.add_argument("--log-level", default="i", type=str, choices=choices, help=f"Set the logging level ({choices})")

#         cli_args = parser.parse_args()

#         # Resolve log_level from args dynamically
#         LOG_LEVEL = levels[next(iter([k for k, v in levels.items() if getattr(cli_args, v.lower(), False)]), cli_args.log_level)]
#     else:
#         LOG_LEVEL = _DEBUG

#     logger_obj = assign_logging_to()
#     try:
#         from importlib import metadata

#         __version__ = metadata.version("nnll")
#     except metadata.PackageNotFoundError as error_log:
#         print(f"nnll package is not installed. Did you run `pip install .`? {error_log}")

# else:
#     from logging import DEBUG as _DEBUG

#     LOG_LEVEL = _DEBUG
#     # if "pytest" not in sys_modules:
#     # logger_obj = assign_logging_to()
