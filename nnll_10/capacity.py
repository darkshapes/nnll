# import os
# import json
# import psutil
# from collections import defaultdict
# from platform import system
# from sdbx.config import config, cache, logging, config_source_location
# import torch


# class SystemCapacity:

#     @cache
#     def write_capacity(self):

#         spec = defaultdict(dict)
#         spec["data"]["dynamo"]  = False if system().lower() == "windows" else True
#         spec["data"]["devices"] = {}
#         if torch.cuda.is_available():
#             spec["data"]["devices"]["cuda"] = torch.cuda.mem_get_info()[1]
#             spec["data"]["flash_attention"] = torch.backends.cuda.flash_sdp_enabled() if system().lower() == "linux" else False
#             spec["data"]["allow_tf32"]      = False
#             spec["data"]["xformers"]        = torch.backends.cuda.mem_efficient_sdp_enabled()
#             if "True" in [spec["data"].get("xformers"), spec["data"].get("flash_attention")]:
#                 spec["data"]["enable_attention_slicing"] = False
#         if torch.backends.mps.is_available() & torch.backends.mps.is_built():
#             spec["data"]["devices"]["mps"] = torch.mps.driver_allocated_memory()
#             try:
#                 import flash_attn
#             except:
#                 spec["data"]["flash_attention"]          = False
#                 spec["data"]["attention_slicing"] = True
#             else:
#                 spec["data"]["flash_attention"] = True  # hope for the best that user set this up
#             #set USE_FLASH_ATTENTION=1 in console
#             # ? https                       : //pytorch.org/docs/master/notes/mps.html
#             # ? memory_fraction = 0.5  https: //iifx.dev/docs/pytorch/generated/torch.mps.set_per_process_memory_fraction
#             # ? torch.mps.set_per_process_memory_fraction(memory_fraction)
#         if torch.xpu.is_available():
#             # todo: code for xpu total memory, possibly code for mkl
#             """ spec["data"]["devices"]["xps"] = ram"""
#         spec["data"]["devices"]["cpu"] = psutil.virtual_memory().total # set all floats = fp32
#         spec_file = os.path.join(config_source_location, "spec.json")
#         if os.path.exists(spec_file):
#             try:
#                 os.remove(spec_file)
#             except FileNotFoundError as error_log:
#                 logging.debug(f"'Spec file absent at write time: {spec_file}.'{error_log}", exc_info=True)
#         if spec:
#             try:
#                 with open(spec_file, "w+", encoding="utf8") as file_out:
#                     """ try opening file"""
#             except Exception as error_log:
#                 logging.debug(f"Error writing spec file '{spec_file}': {error_log}", exc_info=True)
#             else:
#                 with open(spec_file, "w+", encoding="utf8") as file_out:
#                     json.dump(spec, file_out, ensure_ascii=False, indent=4, sort_keys=False)
#         else:
#             logging.debug("No data to write to spec file.", exc_info=True)
#         #return data

#     @cache
#     def get_capacity(self):
#         self.system           = config.get_default("spec","data") #needs to be set by system @ launch
#         system = self.system
#         capacity = {
#             "devices"          : system.get("devices",0),
#             "flash_attention"  : system.get("flash_attention",0),
#             "xformers"         : system.get("xformers",0),
#             "dynamo"           : system.get("dynamo",0),
#             "device"           : next(iter(system.get("devices",0))),
#             "tf32"             : system.get("allow_tf32",0),
#             "attention_slicing": system.get("attention_slicing",0),
#         }

#         return capacity


# @cached_property
# def model_indexer(self) -> Callable:
#     """Model detection and recognition"""
#     from sdbx.indexer import IndexManager

#     return IndexManager()

# @cached_property
# def t2i_pipe(self):
#     """Functionality for text-to-image generation"""  # will probably be moved to nnll
#     from sdbx.nodes.compute import T2IPipe

#     return T2IPipe()

# @cached_property
# def node_tuner(self):
#     """Runtime inference performance optimizations"""
#     from sdbx.nodes.tuner import NodeTuner

#     return NodeTuner()

# def system_profiler(self) -> None:
#     """Collect system GPU, CPU and environment data, then write to file"""
#     from collections import defaultdict
#     from platform import system

#     import psutil

#     spec = defaultdict(dict)
#     spec["data"]["dynamo"] = False if system().lower() == "windows" else True
#     spec["data"]["devices"] = {}
#     if torch.cuda.is_available():
#         spec["data"]["devices"]["cuda"] = torch.cuda.mem_get_info()[1]
#         spec["data"]["flash_attention"] = False  # str(torch.backends.cuda.flash_sdp_enabled()).title()
#         spec["data"]["allow_tf32"] = False
#         spec["data"]["xformers"] = torch.backends.cuda.mem_efficient_sdp_enabled()
#         if "True" in [spec["data"].get("xformers"), spec["data"].get("flash_attention")]:
#             spec["data"]["enable_attention_slicing"] = False
#     if torch.backends.mps.is_available() & torch.backends.mps.is_built():
#         spec["data"]["devices"]["mps"] = torch.mps.driver_allocated_memory()
#         try:
#             import flash_attn  # noqa # pylint: disable=unused-import

#             spec["data"]["flash_attention"] = True  # hope for the best that user set this up
#         except (ImportError, ModuleNotFoundError):
#             spec["data"]["flash_attention"] = False
#             spec["data"]["enable_attention_slicing"] = True
#         # set USE_FLASH_ATTENTION=1 in console
#         # ? https                       : //pytorch.org/docs/master/notes/mps.html
#         # ? memory_fraction = 0.5  https: //iifx.dev/docs/pytorch/generated/torch.mps.set_per_process_memory_fraction
#         # ? torch.mps.set_per_process_memory_fraction(memory_fraction)
#     # if torch.xpu.is_available():
#     #     # add code for xpu total memory, possibly code for mkl
#     #     """ spec["data"]["devices"]["xps"] = ram"""
#     spec["data"]["devices"]["cpu"] = psutil.virtual_memory().total  # set all floats = fp32
#     spec_file = os.path.join(config_source_location, "spec.json")
#     if os.path.exists(spec_file):
#         try:
#             os.remove(spec_file)
#         except (FileNotFoundError, IOError):
#             logging.debug("Spec file absent at write time: %s", spec_file, exc_info=True)
#     if spec:
#         try:
#             with open(spec_file, "w+", encoding="utf8") as file_out:
#                 json.dump(spec, file_out, ensure_ascii=False, indent=4, sort_keys=False)
#         except (json.JSONDecodeError, FileNotFoundError, IOError):
#             logging.debug("Error writing spec file %s", spec_file, exc_info=True)
#     else:
#         logging.debug("No data to write to spec file.", exc_info=True)

