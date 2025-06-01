#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from functools import cache
from pathlib import Path


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


HOME_FOLDER_PATH = set_home_stable()
USER_PATH_NAMED = os.path.join(HOME_FOLDER_PATH, "config.toml")
