### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from tkinter import N
from nnll_01 import debug_monitor


@debug_monitor
def trace_project_structure(walk_path: str) -> list:
    """
    Find all module based on local file/folder structure\n
    :param indicator: Regex for segment of relevant module files or folders
    :return: `list` A list of all module sub-folders with `indicator` in them
    """
    import os
    import re

    ignore_list = ["tests", "log", ".venv", ".git", ".pytest_cache", ".github", "nnll.egg-info"]
    project_directories = []
    for root, _, _ in os.walk(walk_path):
        if any(parent in ignore_list for parent in root.split(os.sep)):
            continue  # for dir in os.path.split(root)
        project_directories.append(root)
    project_directories.sort()
    return project_directories
