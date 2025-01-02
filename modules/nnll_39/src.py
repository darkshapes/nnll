
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


from modules.nnll_30.src import read_json_file
from modules.nnll_46.src import IdConductor
from modules.nnll_47.src import parse_pulled_keys

def filter_header_keys(current_file:str, unpacked_metadata: dict, pattern_reference_path_named="modules/nnll_29/filter.json",) -> dict:
    try:  # Be sure theres something in model_header
        if next(iter(unpacked_metadata),0):
            tensor_count = len(unpacked_metadata)
    except TypeError as error_log:  # Fail if header arrives empty, rather than proceed
        raise error_log
    else:  # Process and output metadata
        pattern_reference  = read_json_file(pattern_reference_path_named)
        conductor_instance = IdConductor()
        conductor_instance.current_file = current_file
        layer_keys         = conductor_instance.identify_layer_type(pattern_reference, unpacked_metadata, tensor_count)
        category_type      = conductor_instance.identify_category_type(layer_keys, pattern_reference, unpacked_metadata, tensor_count)
        model_type         = conductor_instance.identify_model(category_type, pattern_reference, unpacked_metadata, tensor_count)
        pulled_keys        = parse_pulled_keys(layer_keys, category_type, model_type)

        return pulled_keys
