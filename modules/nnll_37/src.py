
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from networkx import connected_components
from tqdm.auto import tqdm
from collections import defaultdict
from contextlib import suppress

from modules.nnll_29.src import LayerFilter
from modules.nnll_30.src import write_json_file
from modules.nnll_32.src import coordinate_header_tools
from modules.nnll_34.src import gather_sharded_files, detect_index_sequence
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

    for current_file in tqdm(folder_contents, total=len(folder_contents), position=0, leave=True):
        file_extension = Path(current_file).suffix.lower()
        file_name = os.path.basename(current_file)
        disk_size = os.path.getsize(current_file)
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

        return (full_model_header, disk_size, file_name, file_extension)

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
    extracted_headers      = collect_file_headers_from(file_or_folder_path_named)

    if extracted_headers is not None:

        metadata_dict = defaultdict(dict)
        model_header, disk_size, file_name, file_extension = extracted_headers
        metadata_dict = {"disk_size": disk_size, "file_name": file_name, "file_extension": file_extension}
        index_tags    = create_model_tag(model_header,disk_size,file_name,file_extension,extracted_headers)
        if index_tags is not None:
            write_json_file(empty_folder_path_to_save_file, "index.json", index_tags, 'w')

# path components
# path list
# current file
