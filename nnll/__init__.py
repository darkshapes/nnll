#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from functools import cache
from pathlib import Path

from nnll.helpers import ensure_path
# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


@cache
def set_home_stable(folder: str = "divisor") -> Path:
    """Retrieve OS specific `App Data` folder path\n
    :param folder: A folder location inside `App Data`
    :return: A platform-specific path to vendor-designated App Data folder\n
    RATIONALE: operator may want to discard the application\n
    EXAMPLES: to maintain experimental conditions, improper venv setup, conflicting dependencies, troubleshooting,
    overreliance on reinstalling to fix things, switching computersquit, disk space full, they got advice online, etc.\n Therefore,
    To accommodate user so they can return to previous settings, leverage os-specific library location.
    """
    from platform import system
    from pathlib import Path

    return (
        str(os.path.join(Path.home(), "AppData", "Local", folder))
        if system().lower() == "windows"
        else str(os.path.join(Path.home(), "Library", "Application Support", folder))
        if system().lower() == "darwin"
        else str(os.path.join(Path.home(), ".config", folder.lower()))
    )


HOME_FOLDER_PATH = ensure_path(folder_path_named=set_home_stable())


def set_log(folder_path_named: str = "log", child: bool = False) -> str:
    """Create logging path\n
    :param folder_path_named: Name of the log folder, defaults to "log"
    :param child: Put log folder inside current working folder, defaults to False
    :return: Path object logging assignment
    """
    prefix = ensure_path(HOME_FOLDER_PATH) if not child else os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(
        prefix,
        folder_path_named,
    )
    return log_folder


USER_PATH_NAMED = ensure_path(Path(os.path.join(HOME_FOLDER_PATH, "config.toml")))
LOG_FOLDER_PATH = ensure_path(Path(set_log()))
