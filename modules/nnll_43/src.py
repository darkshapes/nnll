
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import argparse
import pathlib

from modules.nnll_41.src import trace_file_structure
from modules.nnll_42.src import populate_module_index

def write_toc_to_file(module_index:dict, index_file_name:str='__init__.py', walk_path="modules") -> None:
    write_path = os.path.join(walk_path, index_file_name)
    with open(write_path, 'w') as f: # No +, don't need to read from file
        f.write("#// SPDX-License-Identifier: blessing\n#// d a r k s h a p e s\n\"\"\"\n## module table of contents\n\n")
        for resource, link in module_index.items():
            if ".md" in index_file_name:
                link = f'({resource[1:8]})' # todo : replace with regex
            line_out = "#### " + resource + link + "\n"
            f.write(line_out)
        f.write('"""\n')
    return

def index(folder_name:str = "modules", toc_file:list=["__init__.py","toc.md"], indicator:str='src.py'):
    print('Refreshing index...')
    if not isinstance(toc_file,list):
        [toc_file]
    active_directories = trace_file_structure(folder_name, indicator)
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

