
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from networkx import connected_components
from tqdm.auto import tqdm
from collections import defaultdict
from contextlib import suppress

from modules.nnll_30.src import write_json_file
from modules.nnll_32.src import coordinate_header_tools
from modules.nnll_34.src import gather_sharded_files, detect_index_sequence
from modules.nnll_39.src import filter_header_keys
from modules.nnll_40.src import create_model_tag

def collect_file_headers_from(file_or_folder_path_named: str) -> dict:
    """
    Determine if file or folder path, then extract nn model header metadata at specified location\n
    :param target_file_or_folder: `str` the path to a file or folder to process
    :return: `dict` metadata from the header(s) at the target
    """
    if not os.path.isdir(file_or_folder_path_named):
        folder_contents = [file_or_folder_path_named]
    else:
        folder_contents = file_or_folder_path_named

    model_index = defaultdict(dict)
    #for current_file in tqdm(folder_contents, total=len(folder_contents), position=0, leave=True):
    for current_file in folder_contents:
        file_extension    = Path(current_file).suffix.lower()
        folder_path_named = os.path.dirname(current_file)
        file_name         = os.path.basename(current_file)
        disk_size         = os.path.getsize(current_file)
        disk_path         = os.path.join(folder_path_named,file_name)
        open_header_method = coordinate_header_tools(current_file)
        indexed_file = detect_index_sequence(os.path_basename(current_file))
        if isinstance(indexed_file, tuple):
            gathered_index = gather_sharded_files(current_file, indexed_file)
        else:
            gathered_index = [current_file]
            # Should now be normalized as a list
        full_model_header = defaultdict(dict)
        for next_file in gathered_index:
            next_state_dict = open_header_method(next_file)
            full_model_header.update(next_state_dict)

        if full_model_header is not None:
            pulled_keys = filter_header_keys(full_model_header)
        id_metadata = {"disk_size": disk_size, "disk_path": folder_path_named, "file_extension": file_extension}
        id_metadata.update(pulled_keys)
        index_tag    = create_model_tag(id_metadata)
        model_index.setdefault(disk_path, {index_tag: id_metadata})

    return model_index

def index(file_or_folder_path_named, save):
    with suppress(TypeError):
        if len(sys.argv) != 3:
            print(f"""
Description: Scan specific files or folders for recognized models
Usage: {sys.argv[0]} <folder_path>\n""")
    extracted_headers  = collect_file_headers_from(file_or_folder_path_named)

if __name__ == "__main__":
    file_or_folder_path_named  = "/Users/unauthorized/Downloads/models/text"
    empty_folder_path_to_save_file   = "/Users/unauthorized/Downloads/models/metadata"
    model_index  = collect_file_headers_from(file_or_folder_path_named)
    if model_index is not None:
        write_json_file(empty_folder_path_to_save_file, "index.json", model_index, 'w')