
from collections import defaultdict

import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from modules.nnll_04.src import load_safetensors_metadata
from modules.nnll_30.src import write_json_file

file_name = "/Users/unauthorized/Downloads/models/lora/Hyper-FLUX.1-dev-8steps-lora.safetensors"
# file_name = "/Users/unauthorized/Downloads/models/image/sd_xl_base_1.0.safetensors"
virtual_data_00 = load_safetensors_metadata(file_name)
# print(virtual_data_00)
write_file = os.path.join("/Users/unauthorized/Downloads/models/metadata")
write_json_file(write_file, os.path.basename(file_name), virtual_data_00)
# print(virtual_data_00["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"].get("shape"))  # preferred

final = defaultdict()
# print(virtual_data_00)
for k in virtual_data_00:
    for subkey in virtual_data_00[k]:
        final[subkey] = virtual_data_00[k].get(subkey)
        print(final[subkey])
        if "shape" in subkey:
            shape_data = virtual_data_00[k].get(subkey)
            if shape_data > final.get("shape", 0):
                final["shape"] = shape_data


value = next(reversed(virtual_data_00.values()), None) if virtual_data_00 else None
print(value.get("shape") == final["shape"])
