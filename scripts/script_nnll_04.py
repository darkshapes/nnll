
from collections import defaultdict

import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from modules.nnll_04.src import load_safetensors_metadata
from modules.nnll_30.src import write_json_file

file_name = "/Users/unauthorized/Downloads/models/image/flux1-dev.safetensors"
virtual_data_00 = load_safetensors_metadata(file_name)
write_file = os.path.join("/Users/unauthorized/Downloads/models/metadata")
write_json_file(write_file, os.path.basename(file_name), virtual_data_00)
print(virtual_data_00)
