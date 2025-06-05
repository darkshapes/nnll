#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
from functools import cache
from pathlib import Path
from typing import Optional

# https://huggingface.co/models?library=diffusers


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


def ensure_path(
    folder_path_named: Path = set_home_stable(),
    file_name: Optional[str] = None,
):
    """Provide absolute certainty a file location exists\n
    :param folder_path_named: Location to test, defaults to os.path.dirname(HOME_FOLDER_PATH)
    :param file_name:Optional file name to test, defaults to None
    :return: _description_
    """
    if not Path(folder_path_named).exists():
        try:
            folder_path_named.mkdir(parents=True, exist_ok=True)  # Ensure the directory is created resiliently
        except OSError:
            try:
                os.makedirs(folder_path_named, exist_ok=False)
            except OSError:
                return None

    if file_name:
        full_path = os.path.join(folder_path_named, file_name)
        if Path(full_path).exists():
            return full_path
        try:
            full_path.touch(exist_ok=False)  # Create the file only if it doesn't exist
        except (FileExistsError, OSError):
            pass
        return str(full_path)

    return folder_path_named if Path(folder_path_named).exists() else None


HOME_FOLDER_PATH = ensure_path()


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
