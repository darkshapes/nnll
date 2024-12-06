
from collections import defaultdict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from modules.nnll_05.src import load_gguf_metadata

id_values_00 = defaultdict(dict)

file_name = "/Users/unauthorized/.ollama/models/blobs/sha256-7e0ba38cf5a41b7d1857c2352d3bda2a42e7e24ce7ee0c84e7c242a79781bcd2"
virtual_data_00 = load_gguf_metadata(file_name)
print(virtual_data_00)  # preferred
