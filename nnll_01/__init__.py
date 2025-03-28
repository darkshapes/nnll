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


def configure_logging(file_name: str = ".nnll", folder_path_named: str = "log", time_format: str = "%H:%M:%S.%f") -> Logger:
    """
    Configure and launch *structured* logger integration
    base python logger + formatter + custom logger\n
    :param file_name: The desired filename of the log file
    :param folder_path_named: The desired path of the folder
    :param time_format: The desired time signature format of the log entry
    :return: a `logging` object ready to use for tracking operations
    """
    from datetime import datetime

    import logging
    import structlog
    import litellm
    from structlog.processors import ExceptionPrettyPrinter, StackInfoRenderer, format_exc_info, dict_tracebacks, TimeStamper, JSONRenderer
    from structlog.stdlib import add_log_level, PositionalArgumentsFormatter
    from structlog import configure as structlog_conf, make_filtering_bound_logger, WriteLoggerFactory, get_logger

    file_name += f"{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(folder_path_named, exist_ok=True)
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
    litellm.disable_streaming_logging = True
    litellm.turn_off_message_logging = True
    litellm.suppress_debug_info = False
    litellm.json_logs = True
    handler = logging.FileHandler(assembled_path)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
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
            logger_obj.debug(
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
        except Exception as e:
            logger_obj.debug(
                str(e)
                # exc_info=str(e),
                # filename=func.__module__,
                # pathname=Path(func.__module__).cwd(),
                # func_name=func.__name__,
                # process=os.getppid(),
                # thread=get_native_id(),
                # **{"ain_type": type(args), "ain": args} if args else {},
                # **{"kin_type": type(kwargs), "kin": kwargs} if kwargs else {},
            )
            # Re-raise the exception to propagate it
            raise e

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
