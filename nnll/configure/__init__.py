#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from functools import cache
from pathlib import Path
from sys import argv as sys_argv, modules as sys_modules
from typing import Optional


@cache
def set_home_stable(folder: str = "Shadowbox") -> Path:
    """Retrieve OS specific `App Data` folder path\n
    :param folder: A folder location inside `App Data`
    :return: A platform-specific path to vendor-designated App Data folder\n
    RATIONALE: operator may want to discard the application\n
    EXAMPLES: to maintain experimental conditions, improper venv setup, conflicting dependencies, troubleshooting,
    overreliance on reinstalling to fix things, switching computersquit, disk space full, they got advice online, etc.\n Therefore,
    To accommodate user so they can return to previous settings, leverage os-specific library location.
    """
    from platform import system

    return (
        os.path.join(os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local")), folder)
        if system().lower() == "windows"
        else os.path.join(os.path.expanduser("~"), "Library", "Application Support", folder)
        if system().lower() == "darwin"
        else os.path.join(os.path.expanduser("~"), ".config", folder.lower())
    )


def set_log(folder_path_named: str = "log", child: bool = False) -> str:
    """Create logging path\n
    :param folder_path_named: Name of the log folder, defaults to "log"
    :param child: Put log folder inside current working folder, defaults to False
    :return: Path object logging assignment
    """
    prefix = HOME_FOLDER_PATH if not child else os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(
        prefix,
        folder_path_named,
    )
    return log_folder


HOME_FOLDER_PATH = set_home_stable()


def ensure_path(
    folder_path_named: Path = os.path.dirname(HOME_FOLDER_PATH),
    file_name: Optional[str] = None,
):
    """Provide absolute certainty a file location exists\n
    :param folder_path_named: Location to test, defaults to os.path.dirname(HOME_FOLDER_PATH)
    :param file_name:Optional file name to test, defaults to None
    :return: _description_
    """
    if not os.path.exists(folder_path_named):
        os.makedirs(folder_path_named, exist_ok=False)
    if file_name:
        full_path = os.path.join(folder_path_named, file_name)
        print(full_path)
        if not os.path.exists(full_path):
            with open(full_path, mode="x"):
                pass  # pylint:disable=unspecified-encoding
        return full_path
    return folder_path_named


USER_PATH_NAMED = os.path.join(HOME_FOLDER_PATH, "config.toml")
LOG_FOLDER_PATH = set_log()
