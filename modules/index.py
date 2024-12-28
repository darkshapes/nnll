
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import pathlib
from collections import defaultdict
import re
import argparse


def trace_catalog_file_structure(walk_path: str, indicator:str) -> list:
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

def populate_module_index(active_directories: list, indicator:str) -> dict:
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
            object_name = [re.search(r'\b(?:def|class)\s+(?P<title>\w+)', line)
                    for line in source_code if "__init__" not in line and "main" not in line]
            if object_name:

                # Set format as markup link - '[nnll_## - function_names, etc_etc](link_to_absolute_path)'
                key   = f'[{catalog_number[-1]} - {", ".join({obj.group('title') for obj in object_name if obj is not None})}]'
                value = f'({input_file})'

                # Add or append the value to the index
                if value not in module_index.get(key, module_index.setdefault(key, value)):
                    module_index[key] = module_index[key] + value
                    cat_numbers.append(catalog_number[-1])

    return module_index

def write_toc_to_file(module_index:dict, index_file_name:str='__init__.py', walk_path="modules") -> None:
    write_path = os.path.join(walk_path, index_file_name)
    with open(write_path, 'w') as f: # No +, don't need to read from file
        f.write("#// SPDX-License-Identifier: blessing\n#// d a r k s h a p e s\n\"\"\"\n## module table of contents\n\n")
        for resource, link in module_index.items():
            if ".md" in index_file_name:
                link = f'({resource[1:8]})'
                line_out = "#### " + resource + link + "\n"
            else:
                line_out = "#### " + resource + link + "\n"
            f.write(line_out)
        f.write('"""\n')
    return

def index(folder_name:str = "modules", toc_file:list=["__init__.py","toc.md"], indicator:str='src.py'):
    print('Refreshing index...')
    if not isinstance(toc_file,list):
        [toc_file]
    active_directories = trace_catalog_file_structure(folder_name, indicator)
    module_index       = populate_module_index(active_directories, indicator)
    for file_path_named in toc_file:
        write_toc_to_file(module_index, file_path_named, folder_name)
    module_index.keys()
    print('Done.')

def main():
    parser = argparse.ArgumentParser(
        description="Find all [indicator] within [folder], then write [toc] table of contents in [folder].",
        epilog="Example: nnll-contents src.py modules ['__init__.py', 'toc.md']"
    )
    parser.add_argument(
        '-i', '--indicator', nargs='?', help="Filename convention to be indexed",  const='src.py', default='src.py'
        )
    parser.add_argument(
        '-f', '--folder', nargs='?', help="Path to ouput and root scan folder (default: os.path.join(os.getcwd(), folder_name 'modules')", const="os.path.join(os.getcwd(), 'modules')",  default=os.path.join(os.getcwd(), 'modules')
        )
    parser.add_argument(
        '-t', '--toc','--contents', help="Output files to create (default: ['__init__.py', 'toc.md'])", nargs='*', default=["__init__.py", "toc.md"]
    )
    args = parser.parse_args()

    index(args.folder, args.toc, args.indicator)

if __name__ == "__main__":
    main()

