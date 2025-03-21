### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def trace_file_structure(walk_path: str, indicator=r".*nnll.*") -> list:
    """
    Find all module based on local file/folder structure\n
    :param indicator: Regex for segment of relevant module files or folders
    :return: `list` A list of all module sub-folders with `indicator` in them
    """
    import os
    import re

    indicator_pattern = re.compile(indicator)
    active_directories = []
    for root, directories, files in os.walk(walk_path):
        for path in directories:
            if indicator_pattern.match(path) or [any(indicator_pattern.match(f)) for f in files]:
                active_directories.append(os.path.join(root, path))
    active_directories.sort()
    print(active_directories)
    return active_directories
