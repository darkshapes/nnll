
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from networkx import connected_components
from sympy import pretty_print
from tqdm.auto import tqdm
from collections import defaultdict
from contextlib import suppress
import argparse

from modules.nnll_30.src import write_json_file
from modules.nnll_32.src import coordinate_header_tools
from modules.nnll_34.src import gather_sharded_files, detect_index_sequence
from modules.nnll_39.src import filter_header_keys
from modules.nnll_40.src import create_model_tag
from modules.nnll_27.src import pretty_tabled_output

def collect_file_headers_from(file_or_folder_path_named: str) -> dict:
    """
    Determine if file or folder path, then extract nn model header metadata at specified location\n
    :param target_file_or_folder: `str` the path to a file or folder to process
    :return: `dict` metadata from the header(s) at the target
    """
    if not os.path.isdir(file_or_folder_path_named):
        folder_contents = [file_or_folder_path_named]
    else:
        folder_contents = os.listdir(file_or_folder_path_named)
        folder_path_named = file_or_folder_path_named

    model_index = defaultdict(dict)
    print('\n\n\n\n')
    for current_file in tqdm(folder_contents, total=len(folder_contents), position=0, leave=True):
    # for current_file in folder_contents:
        file_extension     = Path(current_file).suffix.lower()
        valid_extensions = [".sft",".safetensors",".ckpt",".pth",".pt",".gguf"]
        if file_extension == '' or file_extension not in valid_extensions:
            continue
        file_path_named    = os.path.join(folder_path_named,current_file)
        disk_size          = os.path.getsize(file_path_named)
        open_header_method = coordinate_header_tools(file_path_named, file_extension)
        indexed_file       = detect_index_sequence(file_path_named)
        if isinstance(indexed_file, tuple):
            gathered_index = gather_sharded_files(file_path_named, indexed_file)
        else:
            gathered_index = [file_path_named]
            # Should now be normalized as a list
        full_model_header = {}
        for next_file in gathered_index:

            next_state_dict = open_header_method(next_file)
            full_model_header.update(next_state_dict)

        if full_model_header is not None:
            pulled_keys = filter_header_keys(current_file, full_model_header)
            id_metadata = {"disk_size": disk_size, "file_extension": file_extension, "disk_path": folder_path_named, }
            pulled_keys.update(id_metadata)
            index_tag    = create_model_tag(pulled_keys)
            pretty_tabled_output(current_file,pulled_keys)
            model_index.setdefault(file_path_named, index_tag)

    return model_index

def index():
    default_index_folder = os.path.expanduser('~/Downloads/models/image')
    default_save_folder = os.path.expanduser('~/Downloads/models/metadata')
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Identify model(s) at [path], then write their ids to a registry within a json file at [save].",
        epilog="Example: nnll-index ~/Downloads/models/images ~Downloads/models/metadata"
    )
    parser.add_argument(
        "index_folder", nargs='?', help="Path to directory where files should be indexed. (default ' ~/Downloads/models/image')",  const=default_index_folder, default=default_index_folder
    )
    parser.add_argument(
        "save_folder", nargs='?', help="Path where output should be stored. (default: ~/Downloads/models/metadata)", const=default_save_folder, default=default_save_folder
    )
    args = parser.parse_args()
    model_index  = collect_file_headers_from(args.index_folder)
    if model_index is not None:
        write_json_file(args.save_folder, "index.json", model_index, 'w')

if __name__ == "__main__":
    index()