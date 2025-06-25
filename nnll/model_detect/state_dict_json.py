### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
import sys
import os

sys.path.append(os.getcwd())
from nnll.monitor.file import debug_monitor


@debug_monitor
def read_state_dict_headers(folder_path_named: str = ".", save_location: str = ".") -> None:
    """
    Output the full state dict from a model's header to the console and a JSON file.\n
    :param folder_path_named: `str` The location of a model file to read.
    :param save_location: `str` The full path to reserve for output. Must include a `.json` file name.
    :return: `None`
    """
    from nnll.metadata.json_io import write_json_file
    from nnll.metadata.model_tags import ReadModelTags
    from pathlib import Path
    import os

    model_tool = ReadModelTags()
    if folder_path_named is not None:
        for file_name in os.listdir(folder_path_named):
            file = os.path.join(folder_path_named, file_name)
            virtual_data_00 = model_tool.read_metadata_from(file)
            if virtual_data_00 is not None:
                write_json_file(save_location, f"{file_name}.json", virtual_data_00)


@debug_monitor
def main():
    """Parse arguments to feed to dict header reader"""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Output state dict from a model file at [path] to the console, then write to a json file at [save].", epilog="Example: nnll-parse ~/Downloads/models/images ~Downloads/models/metadata")
    parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=".")
    parser.add_argument("save", help="Path where output should be stored. (default: current directory)", default=".")
    args = parser.parse_args()

    read_state_dict_headers(args.path, args.save)


if __name__ == "__main__":
    main()
