### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""初始化 init, argparse, 建立控制檯日誌 logging"""

# pylint: disable=line-too-long, invalid-name, import-outside-toplevel
# noqa: F401
# pylint:disable=ungrouped-imports

import os
from argparse import ArgumentParser
from logging import DEBUG, INFO, Logger  # noqa: F401, pylint:disable=unused-import
from pathlib import Path
from typing import Callable, Literal
from threading import get_native_id
from datetime import datetime
from sys import modules as sys_modules
from nnll.configure import LOG_FOLDER_PATH

EXC_INFO = any(mod in sys_modules for mod in ["textual"] if "pytest" not in sys_modules)


def configure_logging(file_name: str = ".nnll", folder_path_named: str = LOG_FOLDER_PATH, time_format: str = "%H:%M:%S.%f", level: str | Literal[10] = DEBUG) -> Logger:
    """
    Configure and launch *structured* logger integration
    base python logger + formatter + custom logger\n
    :param file_name: The desired filename of the log file
    :param folder_path_named: The desired path of the folder
    :param time_format: The desired time signature format of the log entry
    :return: a `logging` object ready to use for tracking operations
    """

    import logging
    import structlog

    from structlog.processors import ExceptionPrettyPrinter, StackInfoRenderer, format_exc_info, dict_tracebacks, TimeStamper, JSONRenderer
    from structlog.stdlib import add_log_level, PositionalArgumentsFormatter
    from structlog import configure as structlog_conf, make_filtering_bound_logger, WriteLoggerFactory, get_logger

    file_name += f"{datetime.now().strftime('%Y%m%d')}"
    assembled_path = os.path.join(folder_path_named, file_name)

    timestamper = [  # replicate in both loggers
        TimeStamper(fmt=time_format, key="+ts", utc=True),
        add_log_level,
        PositionalArgumentsFormatter(),
        ExceptionPrettyPrinter(),
        StackInfoRenderer(),
        format_exc_info,
        dict_tracebacks,
    ]

    structlog_conf(
        processors=timestamper
        + [
            JSONRenderer(
                indent=1,
                sort_keys=True,
            ),
        ],
        wrapper_class=make_filtering_bound_logger(min_level=0),
        logger_factory=WriteLoggerFactory(file=Path(assembled_path).open("at", encoding="utf-8")),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=timestamper,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(
                indent=1,
                sort_keys=True,
            ),
        ],
    )

    handler = logging.FileHandler(assembled_path)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    # litellm._turn_on_debug()
    # logging.root.addHandler(handler)

    logger = get_logger()
    return logger


def debug_monitor(func: Callable = None) -> Callable:
    """Debug output decorator function
    Data returned from decorated methods/functions is automatically sent to debugger
    """

    def wrapper(*args, **kwargs) -> None:
        """Wrap log"""
        try:
            return_data = func(*args, **kwargs)
            DBUG_OBJ.debug(
                {
                    str(return_data): {
                        "filename": func.__module__,
                        "pathname": Path(func.__module__).cwd(),
                        "func_name": func.__name__,
                        "process": os.getppid(),
                        "thread": get_native_id(),
                        **{"ain_type": type(args), "ain": args if args else {}},
                        **{"kin_type": type(kwargs), "kin": kwargs if kwargs else {}},
                    }
                }
            )

            return return_data
        except Exception as error_log:
            raise error_log

    return wrapper


def info_stream():
    """info console logging\n
    :return: INFO level logging object
    """
    from rich.console import Console
    from rich.logging import RichHandler
    from logging import StreamHandler, Formatter, getLogger
    from nnll.configure.color_map import grey_nouveau_theme

    # from logging import root
    from sys import stderr as sys_stderr

    console_out = Console(stderr=True, theme=grey_nouveau_theme())
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


INFO_OBJ = info_stream()


def nfo(*args, **kwargs) -> None:  # pylint:disable=unused-argument
    """Info log output"""
    try:
        INFO_OBJ.info("%s", f"{args}")
    except ImportError:
        pass


def dbug(*args, **kwargs) -> None:
    """Info log output"""
    try:
        DBUG_OBJ.debug(f"{args, kwargs}", type_ain=type(args), ain=args, type_kin=type(kwargs), kin=kwargs, stack_info=True, exc_info=EXC_INFO)
    except ImportError:
        pass


if __name__ == "__main__":
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

    DBUG_OBJ = configure_logging(level=LOG_LEVEL)

    try:
        from importlib import metadata

        __version__ = metadata.version("nnll")
    except metadata.PackageNotFoundError as error_log:
        dbug(f"nnll package is not installed. Did you run `pip install .`? {error_log}", tb=error_log.__traceback__)
else:
    LOG_LEVEL = DEBUG
    DBUG_OBJ = configure_logging()
