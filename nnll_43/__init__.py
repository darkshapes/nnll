### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def write_toc_to_file(module_index: dict, index_file_name: str, root_folder: str) -> None:
    """Output module data to .md file
    :param module_index: Discovered modules
    :param index_file_name: File to write to, defaults to "__init__.py"
    :param root_folder: _description_, defaults to "modules"
    """
    import os

    write_path = os.path.join(root_folder, index_file_name)
    with open(write_path, "w", encoding="utf-8") as toc_file:  # No +, don't need to read from file
        toc_file.write('#// SPDX-License-Identifier: LAL-1.3\n#// d a r k s h a p e s\n"""\n## module table of contents\n\n')
        for resource, link in module_index.items():
            if ".md" in index_file_name:
                link = os.path.dirname(resource)
            line_out = "#### " + resource + link + "\n"
            toc_file.write(line_out)
        toc_file.write('"""\n')
    return


def index(
    file_indicator: list | str = None,
    root_folder: str = None,
    subfolder_indicator: str = "nnll_",
    toc_file: list | str = None,
):
    """Provide indexing operation for package
    :param indicator: _description_, defaults to ".py"
    :param folder_name: _description_, defaults to "modules"
    :param toc_file: _description_, defaults to None

    """
    from nnll_41 import trace_project_structure
    from nnll_42 import populate_module_index
    from pprint import pprint

    print("Refreshing index...")
    if not file_indicator:
        file_indicator = ["__init__*", "__main__.*"]
    active_directories = trace_project_structure(root_folder)
    module_index = populate_module_index(active_directories, file_indicator)
    if toc_file is not None:
        if not isinstance(toc_file, list):
            toc_file = [toc_file]
        for file_path_named in toc_file:
            write_toc_to_file(module_index, file_path_named, root_folder)
    pprint([f"{func}" for func in module_index.keys()], width=150, compact=True, sort_dicts=True)


def main():
    import argparse
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(current_dir)
    folder_names = "nnll_"

    parser = argparse.ArgumentParser(
        description="Find all [indicators] within [folder]. Optionally write a [toc] table of contents in [folder].",
        epilog="Example: nnll-contents src.py -t ['__init__.py', 'README.md']",
    )
    parser.add_argument(
        "-i",
        "--indicator",
        nargs="?",
        help="Regex pattern of filename convention(s) to be indexed (default: r'__init__\.py|main\.py)",
        const=["__init__.py", "__main__.py"],
        default=["__init__.py", "__main__.py"],
    )
    parser.add_argument(
        "-r",
        "--root",
        nargs="?",
        help="Path to root scan folder (default: parent folder of launch file)",
        const=f"{root_path}",
        default=f"{root_path}",
    )
    parser.add_argument(
        "-f",
        "--folder",
        nargs="?",
        help="List of folder convention(s) to be indexed",
        const=folder_names,
        default=folder_names,
    )
    parser.add_argument(
        "-t",
        "--toc",
        "--contents",
        help="Output files to create (default: None)",
        nargs="?",
        const=None,
        default=None,
    )

    args = parser.parse_args()
    index(file_indicator=args.indicator, root_folder=args.root, subfolder_indicator=args.folder, toc_file=args.toc)


if __name__ == "__main__":
    main()
