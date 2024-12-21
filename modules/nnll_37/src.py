#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s

import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

from modules.nnll_32.src import get_model_header
from modules.nnll_07.src import Domain, Architecture, Component
from modules.nnll_27.src import pretty_tabled_output
from modules.nnll_29.src import LayerFilter
from modules.nnll_30.src import read_json_file, write_json_file
from modules.nnll_34.src import preprocess_files

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

def prepare_tags(disk_path: str) -> None: #this is a full path
    file_paths_shard_linked = preprocess_files(disk_path)
    print("\n\n\n")
    for each_file in tqdm(file_paths_shard_linked, total=len(file_paths_shard_linked), position=0, leave=True):
        data = get_model_header(each_file)  # save_location)
        if data is not None:
            model_header, disk_size, file_name, file_extension = data
        else:
            return
        parse_file = parse_model_header(model_header)
        reconstructed_file_path = os.path.join(disk_path,each_file)
        attribute_dict = {"disk_size": disk_size, "disk_path": reconstructed_file_path, "file_name": file_name, "file_extension": file_extension}
        file_metadata = parse_file | attribute_dict
        index_tag = create_model_tag(file_metadata)
        try:
            pretty_tabled_output(next(iter(index_tag)), index_tag[next(iter(index_tag))])  # output information
        except TypeError as errorlog:
            raise
    return index_tag


disk_path = "/Users/unauthorized/Downloads/models/text"
save_location = "/Users/unauthorized/Downloads/models/metadata"
index_tags = defaultdict(dict)
index_tags = prepare_tags(disk_path)
if index_tags is not None:
    write_json_file(save_location, "index.json", index_tags, 'w')
