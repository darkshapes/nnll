### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def read_state_dict_headers(folder_path_named: str = ".", save_location: str = ".") -> None:
    """
    Output the full state dict from a model's header to the console and a JSON file.\n
    :param folder_path_named: `str` The location of a model file to read.
    :param save_location: `str` The full path to reserve for output. Must include a `.json` file name.
    :return: `None`
    """
    from nnll_30 import write_json_file
    from nnll_32 import coordinate_header_tools
    from pathlib import Path
    import os

    if folder_path_named is not None:
        for file_name in os.listdir(folder_path_named):
            # if file_name
            file = os.path.join(folder_path_named, file_name)
            extension = Path(file_name).suffix
            header_method = coordinate_header_tools(file, extension)
            if header_method is not None:
                virtual_data_00 = header_method(file)
                if virtual_data_00 is not None:
                    write_json_file(save_location, f"{file_name}.json", virtual_data_00)


def main():
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Output state dict from a model file at [path] to the console, then write to a json file at [save].", epilog="Example: nnll-parse ~/Downloads/models/images ~Downloads/models/metadata")
    parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=".")
    parser.add_argument("save", help="Path where output should be stored. (default: current directory)", default=".")
    args = parser.parse_args()

    read_state_dict_headers(args.path, args.save)


if __name__ == "__main__":
    main()
