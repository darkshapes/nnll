### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""初始化 init, argparse, 建立控制檯日誌 logging"""

# pylint: disable=line-too-long, invalid-name, import-outside-toplevel
# noqa: F401, pylint:disable=ungrouped-imports

import os
from argparse import ArgumentParser
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Logger  # noqa: F401, pylint:disable=unused-import
from pathlib import Path
from typing import Callable
from threading import get_native_id


def use_nouveau_theme():
    """Set a a midnight blue and dark purple theme to console logs"""
    from rich import style
    from rich.theme import Theme

    NOUVEAU: Theme = Theme(
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
    return NOUVEAU


def configure_logging(file_name: str = ".nnll", folder_path_named: str = "log", time_format: str = "%H:%M:%S") -> Logger:
    """
    Configure and launch logger\n
    :param file_name: The desired filename of the log file
    :param folder_path_named: The desired path of the folder
    :param time_format: The desired time signature format of the log entry
    :return: a `logging` object ready to use for tracking operations
    """
    from datetime import datetime
    from logging import basicConfig  # noqa: F401, pylint:disable=unused-import

    from structlog import WriteLoggerFactory, get_logger, make_filtering_bound_logger
    from structlog import configure as sl_conf
    from structlog.processors import ExceptionPrettyPrinter, JSONRenderer, StackInfoRenderer, TimeStamper, format_exc_info, CallsiteParameter, CallsiteParameterAdder

    file_name += f"{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(folder_path_named, exist_ok=True)
    assembled_path = os.path.join(folder_path_named, file_name)
    # basicConfig(format="%(message)s", datefmt=time_format, level=DEBUG)

    sl_conf(
        cache_logger_on_first_use=True,
        wrapper_class=make_filtering_bound_logger(DEBUG),
        processors=[
            CallsiteParameterAdder(
                parameters={
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    # CallsiteParameter.LINENO,
                    # CallsiteParameter.MODULE,
                    CallsiteParameter.PATHNAME,
                    CallsiteParameter.PROCESS,
                    CallsiteParameter.THREAD,
                }
            ),
            TimeStamper(fmt=time_format, key="+ts"),
            StackInfoRenderer(),
            format_exc_info,
            ExceptionPrettyPrinter(),
            JSONRenderer(
                indent=1,
                sort_keys=True,
            ),
        ],
        logger_factory=WriteLoggerFactory(file=Path(assembled_path).open("at", encoding="utf-8")),
    )
    logger = get_logger(__name__)  # snake case is intentional
    return logger


def debug_monitor(func: Callable = None) -> Callable:
    """Debug output decorator function
    Data returned from decorated methods/functions is automatically sent to debugger
    """

    def wrapper(*args, **kwargs) -> None:
        """Wrap log"""
        return_data = func(*args, **kwargs)
        if not kwargs:
            logger_obj.debug(
                filename=func.__module__,
                pathname=Path(func.__module__).cwd(),
                func_name=func.__name__,
                process=os.getpid(),
                thread=get_native_id(),
                event={
                    "ain_type": type(args),
                    "ain": args,
                    "output": return_data,
                },
            )
        else:
            logger_obj.debug(
                filename=func.__module__,
                pathname=Path(func.__module__).cwd(),
                func_name=func.__name__,
                process=os.getpid(),
                thread=get_native_id(),
                event={
                    "ain_type": type(args),
                    "ain": args,
                    "kin_type": type(kwargs),
                    "kin": kwargs,
                    "output": return_data,
                },
            )
        return return_data

    return wrapper


def info_message(*args, **kwargs):
    """Info log output"""
    logger_obj.info(
        "%s",
        type_ain=type(args),
        ain=args,
        type_kin=type(kwargs),
        kin=kwargs,
        stack_info=True,
    )


def debug_message(*args, **kwargs):
    """Info log output"""
    logger_obj.debug(
        "%s",
        type_ain=type(args),
        ain=args,
        type_kin=type(kwargs),
        kin=kwargs,
        stack_info=True,
    )


if __name__ == "__main__":
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = ArgumentParser(description="Set logging level.")
        group = parser.add_mutually_exclusive_group()

        levels = {"d": "DEBUG", "w": "WARNING", "e": "ERROR", "c": "CRITICAL", "i": "INFO"}
        choices = list(levels.keys()) + list(levels.values()) + [value.upper() for value in levels.values()]
        for short, long in levels.items():
            group.add_argument(f"-{short}", f"--{long.lower()}", f"--{long}", action="store_true", help=f"Set logging level {long}")

        group.add_argument("--log-level", default="i", type=str, choices=choices, help=f"Set the logging level ({choices})")

        cli_args = parser.parse_args()

        # Resolve log_level from args dynamically
        LOG_LEVEL = levels[next(iter([k for k, v in levels.items() if getattr(cli_args, v.lower(), False)]), cli_args.log_level)]
    else:
        LOG_LEVEL = DEBUG

    logger_obj = configure_logging()
    try:
        from importlib import metadata

        __version__ = metadata.version("nnll")
    except metadata.PackageNotFoundError as error_log:
        debug_message(f"nnll package is not installed. Did you run `pip install .`? {error_log}", tb=error_log.__traceback__)
else:
    LOG_LEVEL = DEBUG
    logger_obj = configure_logging()

    from asyncio import run as asyncio_run
    # alogger_obj = configure_async_logging()
    # asyncio_run(async_monitor())


# def configure_async_logging(file_name: str = ".nnll_async", folder_path_named: str = "log", time_format: str = "%H:%M:%S") -> Logger:
#     from datetime import datetime
#     from structlog import configure as st_conf, WriteLoggerFactory, get_logger
#     from structlog.processors import JSONRenderer, add_log_level, format_exc_info, TimeStamper
#     from structlog.stdlib import AsyncBoundLogger
#     from structlog.contextvars import merge_contextvars

#     file_name += f"{datetime.now().strftime('%Y%m%d')}"
#     os.makedirs(folder_path_named, exist_ok=True)
#     assembled_path = os.path.join(folder_path_named, file_name)

#     st_conf(
#         processors=[
#             TimeStamper(fmt=time_format, key="+ts"),
#             merge_contextvars,
#             add_log_level,
#             format_exc_info,
#             JSONRenderer(),
#         ],
#         wrapper_class=AsyncBoundLogger,
#         context_class=dict,
#         cache_logger_on_first_use=True,
#         logger_factory=WriteLoggerFactory(file=Path(assembled_path).open("at", encoding="utf-8")),
#     )
#     return get_logger()

# async def async_monitor(func: Callable = None) -> Callable:
#     """Debug output decorator function
#     Data returned from decorated methods/functions is automatically sent to debugger
#     """

#     async def wrapper(*args, **kwargs) -> None:
#         """Wrap log"""
#         return_data = func(*args, **kwargs)
#         if not kwargs:
#             await alogger_obj.adebug(
#                 "%s",
#                 filename=func.__module__,
#                 pathname=Path(func.__module__).cwd(),
#                 func_name=func.__name__,
#                 ain_type=type(args),
#                 ain=args,
#                 output=return_data,
#             )
#         else:
#             await alogger_obj.adebug(
#                 "%s",
#                 filename=func.__module__,
#                 pathname=Path(func.__module__).cwd(),
#                 func_name=func.__name__,
#                 ain_type=type(args),
#                 ain=args,
#                 kin_type=type(kwargs),
#                 kin=kwargs,
#                 output=return_data,
#             )
#         return return_data

# return wrapper
