### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os


def trace_file_structure(walk_path: str, indicator: str) -> list:
    """
    Find all module based on local file structure\n
    :param indicator: `str` Name or suffix type of relevant module files
    :return: `list` A list of all module sub-folders with `indicator` in them
    """
    active_directories = []
    for root_folder, _, file_names in os.walk(walk_path, topdown=False):
        if indicator in file_names:
            active_directories.append(root_folder)
    active_directories.sort()
    return active_directories  # Organize by numerical order
