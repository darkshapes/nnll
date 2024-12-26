
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from networkx import connected_components
from tqdm.auto import tqdm
from collections import defaultdict

from modules.nnll_30.src import write_json_file
from modules.nnll_32.src import get_model_header
from modules.nnll_34.src import gather_sharded_files
from modules.nnll_40.src import create_model_tag

def collect_file_headers_from(target_file_or_folder: str) -> dict:
    """
    Determine if file or folder path, then extract nn model header metadata at specified location\n
    :param target_file_or_folder: `str` the path to a file or folder to process
    :return: `dict` metadata from the header(s) at the target
    """

    if not os.path.isdir(target_file_or_folder):
        gathered_file_targets = list(gather_sharded_files(target_file_or_folder))
    else:
        gathered_file_targets = (gather_sharded_files(current_file) for current_file in target_file_or_folder)
    for current_file in tqdm(gathered_file_targets, total=len(gathered_file_targets), position=0, leave=True):
        extracted_headers = get_model_header(current_file)
        return extracted_headers #will this only one run time?

if __name__               == "__main__":
    target_file_or_folder  = "/Users/unauthorized/Downloads/models/text"
    target_save_location   = "/Users/unauthorized/Downloads/models/metadata"
    extracted_headers      = collect_file_headers_from(target_file_or_folder)

    if extracted_headers is not None:
        metadata_dict = defaultdict(dict)
        model_header, disk_size, file_name, file_extension = extracted_headers
        metadata_dict = {"disk_size": disk_size, "file_name": file_name, "file_extension": file_extension}
        index_tags    = create_model_tag(model_header,disk_size,file_name,file_extension,extracted_headers)
        if index_tags is not None:
            write_json_file(target_save_location, "index.json", index_tags, 'w')

# path components
# path list
# current file
