
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from modules.nnll_29.src import LayerFilter
from modules.nnll_30.src import read_json_file, write_json_file

def parse_model_header(model_header: dict, filter_file="modules/nnll_29/filter.json") -> dict:
    try:  # Be sure theres something in model_header
        next(iter(model_header))
    except TypeError as errorlog:
        raise  # Fail if header arrives empty, rather than proceed
    else:  # Process and output metadata
        FILTER = read_json_file(filter_file)
        tensor_count = len(model_header)
        block_scan = LayerFilter()
        file_metadata = block_scan.reference_walk_conductor(FILTER, model_header, tensor_count)
        return file_metadata
