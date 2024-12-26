
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from collections import defaultdict

import os
from pathlib import Path
import argparse

from modules.nnll_04.src import load_safetensors_metadata_from_model
from modules.nnll_05.src import load_gguf_metadata_from_model
from modules.nnll_30.src import write_json_file


def parse_data(disk_path: str, save_location: str) -> None:
    """
    Output the full state dict from a model's header to the console and a JSON file.\n
    :param disk_path: `str` The location of a model file to read.
    :param save_location: `str` The full path to reserve for output. Must include a `.json` file name.
    :return: `None`
    """
    if disk_path is not None:
        for file_name in os.listdir(disk_path):
            # if file_name
            file = os.path.join(disk_path, file_name)
            if Path(file_name).suffix == ".safetensors":
                virtual_data_00 = load_safetensors_metadata_from_model(file)

            elif Path(file_name).suffix == ".gguf":
                virtual_data_00 = load_gguf_metadata_from_model(file)

            if virtual_data_00 is not None:
                print(virtual_data_00)
                write_file = os.path.join(save_location)
                write_json_file(write_file, f"{file_name}.json", virtual_data_00)

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Output state dict from a model file at [path] to the console, then write to a json file at [save].",
        epilog="Example: python script_modules.nnll_04.py ~/Downloads/models/images ~Downloads/models/metadata"
    )
    parser.add_argument(
        "path", help="Path to directory or file to be analyzed. (default .)", default="."
    )
    parser.add_argument(
        "save", help="Location to save a .json of output. (default index.json)", default="index.json"
    )
    args = parser.parse_args()

    parse_data(args.path, args.save)


if __name__ == "__main__":
    main()
