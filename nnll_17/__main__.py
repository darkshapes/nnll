# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->
# q


# def hash_layers(path: str):
#     import os
#     from nnll_04 import ModelTool
#     from nnll_44 import compute_hash_for
#     from nnll_01 import info_message as nfo
#     from pathlib import Path

#     model_tool = ModelTool()
#     nfo(path)
#     nfo("{")
#     for each in os.listdir(os.path.normpath(Path(path))):
#         if Path(each).suffix.lower() == ".safetensors":
#             state_dict = model_tool.read_metadata_from(os.path.join(path, each))
#             hash_value = compute_hash_for(text_stream=str(state_dict))
#             nfo(f"'{hash_value}' : '{each}'")
#     nfo("}")
