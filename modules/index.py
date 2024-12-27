
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import pathlib
from collections import defaultdict
import re

INDICATOR  = 'src.py'
WALK_PATH  = os.path.join(os.getcwd(), "modules")
INDEX_FILE = "__init__.py"
HEADER_TEXT_BLOCK = "#  // SPDX-License-Identifier: blessing\n#  // d a r k s h a p e s\n\"\"\"\n## module_index\n\n"

def trace_catalog_file_structure(indicator:str=INDICATOR, walk_path: str=WALK_PATH) -> list:
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
    return active_directories # Organize by numerical order

def populate_module_index(active_directories: list, indicator:str=INDICATOR) -> dict:
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
            catalog_number = pathlib.Path(folder_location).parts
            source_code = open(input_file)

            # Gather function names in the module as determined by preceding keywords
            object_name = [re.search(r'\b(?:def|class)\s+(?P<title>\w+)', line)
                    for line in source_code if "__init__" not in line and "main" not in line]
            if object_name:

                # Set format as markup link - '[nnll_## - function_names, etc_etc](link_to_absolute_path)'
                key   = f'[{catalog_number[-1]} - {", ".join({obj.group('title') for obj in object_name if obj is not None})}]'
                value = f'({input_file})'

                # Add or append the value to the index
                if value not in module_index.get(key, module_index.setdefault(key, value)):
                    module_index[key] = module_index[key] + value

    return module_index

def write_index_to_file(module_index: dict, index_file_name: str = INDEX_FILE) -> None:
    write_path = os.path.join(WALK_PATH, index_file_name)
    with open(write_path, 'w') as f: # No +, don't need to read from file
        f.write(HEADER_TEXT_BLOCK)
        for resource, link in module_index.items():
            line_out = "#### " + resource + link + "\n"
            f.write(line_out)
        f.write('"""\n')
    return

def index():
    print('Refreshing index...')
    active_directories = trace_catalog_file_structure()
    module_index       = populate_module_index(active_directories)
    write_index_to_file(module_index)
    print('Done.')

if __name__ == "__main__":
    index()

