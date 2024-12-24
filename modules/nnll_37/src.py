#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from networkx import connected_components
from tqdm.auto import tqdm
from collections import defaultdict

from modules.nnll_32.src import get_model_header
from modules.nnll_07.src import Domain, Architecture, Component
from modules.nnll_27.src import pretty_tabled_output
from modules.nnll_29.src import LayerFilter
from modules.nnll_30.src import read_json_file, write_json_file
from modules.nnll_34.src import gather_sharded_files

def parse_model_header(model_header: dict, filter_file="modules/nnll_29/filter.json") -> dict:
    try:  # Be sure theres something in model_header
        next(iter(model_header))
    except TypeError as errorlog:
        raise  # Fail if header arrives empty, rather than proceed
    else:  # Process and output metadata
        FILTER = read_json_file(filter_file)
        tensor_count = len(model_header)
        block_scan = LayerFilter()
        file_metadata = block_scan.filter_metadata(FILTER, model_header, tensor_count)
        return file_metadata


def collect_file_headers_from(target_for_analysis: str) -> dict:
    """
    Determine if file or folder path, then extract header metadata at specified location\n
    :param target_for_analysis: `str` the path to a file or folder to process
    :return: `dict` metadata from the header(s) at the target
    """

    if not os.path.isdir(target_for_analysis):
        files_linked_with_shards = list(gather_sharded_files(target_for_analysis))
    else:
        files_linked_with_shards = (gather_sharded_files(file) for file in target_for_analysis)
    for each_file in tqdm(files_linked_with_shards, total=len(files_linked_with_shards), position=0, leave=True):
        extracted_keys = get_model_header(each_file)
        return data


def create_model_tag(file_metadata: dict) -> dict:

    if "unknown" in file_metadata:
        domain_dev = Domain("dev")  # For unrecognized models,
    else:
        domain_ml = Domain("ml")  # create the domain only when we know its a model

    arch_found = Architecture(file_metadata.get("model"))
    category = file_metadata["category"]
    file_metadata.pop("model")
    file_metadata.pop("category")
    comp_inside = Component(category, **file_metadata)
    arch_found.add_component(comp_inside.model_type, comp_inside)
    domain_ml.add_architecture(arch_found.architecture, arch_found)
    index_tag = domain_ml.to_dict()

    return index_tag

def create_model_tag(model_header,metadata_dict):
        parse_file = parse_model_header(model_header)
        reconstructed_file_path = os.path.join(disk_path,each_file)
        attribute_dict = metadata_dict | {"disk_path": reconstructed_file_path}
        file_metadata = parse_file | attribute_dict
        index_tag = create_model_tag(file_metadata)
        try:
            pretty_tabled_output(next(iter(index_tag)), index_tag[next(iter(index_tag))])  # output information
        except TypeError as errorlog:
            raise
        return index_tag


target_for_analysis = "/Users/unauthorized/Downloads/models/text"
target_save_location = "/Users/unauthorized/Downloads/models/metadata"
index_tags = defaultdict(dict)
index_tags = collect_file_headers_from(target_for_analysis)
# if index_tags is None:
#     return
# else:
#     metadata_dict = defaultdict(dict)
#     model_header, disk_size, file_name, file_extension = data
#     metadata_dict = {"disk_size": disk_size, "file_name": file_name, "file_extension": file_extension}
#     create_model_tag(model_header,disk_size,file_name,file_extension,each_file)
if index_tags is not None:
    write_json_file(target_save_location, "index.json", index_tags, 'w')

# path components
# path list
# current file
