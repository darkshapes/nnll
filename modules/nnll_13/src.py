#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s

# import os
# import json
# import psutil
# from collections import defaultdict
# from platform import system
# import torch

# class SystemCapacity:

#     def detect_capacity(self):
#         spec = defaultdict(dict)
#         spec["data"]["dynamo"]  = False if system().lower() == "windows" else True
#         spec["data"]["devices"] = {}
#         if torch.cuda.is_available():
#             try:
#                 spec["devices"]["xpu"] = torch.xpu.is_available()
#                 spec["devices"]["rocm"] = torch.version.hip
#             except:
#                 pass
#             try:
#                 torch.cuda.get_device_name(torch.device("cuda")).endswith("[ZLUDA]")
#             except:
#                 pass
#             else:
#                 spec["devices"]["zluda"] = True
#             try:
#                 import xformers
#             except:
#                 pass
#             else:
#                 spec["data"]["xformers"]= torch.backends.cuda.mem_efficient_sdp_enabled()
#                 try:
#                     import flash_attn
#                 except:
#                     spec["data"]["flash_attention"] = False
#                     spec["data"]["attention_slicing"] = True
#                 else:
#                     spec["data"]["flash_attention"] = torch.backends.cuda.flash_sdp_enabled() if system().lower() == "linux" else False
#             spec["data"]["allow_tf32"]      = False
#             spec["data"]["devices"]["cuda"] = torch.cuda.mem_get_info()[1]


#         if torch.xpu.is_available():
#             # todo: code for xpu total memory, possibly code for mkl
#             """ spec["data"]["devices"]["xps"] = ram"""
#         elif system().lower() == "darwin" & torch.backends.mps.is_available() & torch.backends.mps.is_built():
#             spec["data"]["devices"]["mps"] = torch.mps.driver_allocated_memory()
#             spec["data"]["attention_slicing"] = True
#             #set USE_FLASH_ATTENTION=1 in console
#             # ? https                       : //pytorch.org/docs/master/notes/mps.html
#             # ? memory_fraction = 0.5  https: //iifx.dev/docs/pytorch/generated/torch.mps.set_per_process_memory_fraction
#             # ? torch.mps.set_per_process_memory_fraction(memory_fraction)
#         else:
#             spec["data"]["devices"]["cpu"] = psutil.virtual_memory().total # set all floats = fp32

#     def optimal_config():
#         torch.backends.cudnn.allow_tf32
