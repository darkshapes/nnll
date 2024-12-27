
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
from modules.nnll_07.src import Domain, Architecture, Component
from modules.nnll_27.src import pretty_tabled_output
from modules.nnll_39.src import parse_model_header


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

# def create_model_tag(model_header,metadata_dict):
#         parse_file = parse_model_header(model_header)
#         reconstructed_file_path = os.path.join(disk_path,each_file)
#         attribute_dict = metadata_dict | {"disk_path": reconstructed_file_path}
#         file_metadata = parse_file | attribute_dict
#         index_tag = create_model_tag(file_metadata)
#         try:
#             pretty_tabled_output(next(iter(index_tag)), index_tag[next(iter(index_tag))])  # output information
#         except TypeError as errorlog:
#             raise
#         return index_tag
