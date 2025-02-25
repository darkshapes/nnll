### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import pathlib
from collections import defaultdict
import re

from modules.nnll_41.src import trace_file_structure


def populate_module_index(active_directories: list, indicator: str) -> dict:
    """
    Generate a page of absolute path links to the catalog modules, including function names\n
    :param active_directories: `list` The folders to search for modules within
    :param indicator: `str` Name or suffix type of relevant module files
    :return:
    """
    module_index = defaultdict(dict)
    for folder_location in active_directories:
        input_file = os.path.join(folder_location, indicator)
        if not os.path.exists(input_file):
            return
        else:
            cat_numbers = []
            catalog_number = pathlib.Path(folder_location).parts
            source_code = open(input_file)

            # Gather function names in the module as determined by preceding keywords
            object_name = [re.search(r"\b(?:def|class)\s+(?P<title>\w+)", line) for line in source_code if "__init__" not in line and "main" not in line]
            if object_name:
                # Set format as markup link - '[nnll_## - function_names, etc_etc](link_to_absolute_path)'
                key = f"[{catalog_number[-1]} - {', '.join({obj.group('title') for obj in object_name if obj is not None})}]"
                value = f"({input_file})"

                # Add or append the value to the index
                if value not in module_index.get(key, module_index.setdefault(key, value)):
                    module_index[key] = module_index[key] + value
                    cat_numbers.append(catalog_number[-1])

    return module_index
