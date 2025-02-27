### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
from nnll_07 import Domain, Architecture, Component
from nnll_27 import pretty_tabled_output
from nnll_39 import route_metadata


# parse metadata into tag?
def create_model_tag(pulled_keys: dict) -> dict:
    if "unknown" in pulled_keys:
        domain_dev = Domain("dev")  # For unrecognized models,
    else:
        domain_ml = Domain("ml")  # create the domain only when we know its a model

    arch_found = Architecture(pulled_keys.get("architecture"))
    model_type = pulled_keys["model_type"]
    pulled_keys.pop("architecture")
    pulled_keys.pop("model_type")
    comp_inside = Component(model_type, **pulled_keys)
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
#
