#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# import os
# from asyncio import taskgroups
# import re
# from typing import Dict, List, Tuple

# from huggingface_hub import scan_cache_dir

# from pydantic import BaseModel, computed_field

# from package.conversion_graph import NX_GRAPH


# class HubRegistryEntry(BaseModel):
#     """Validate Hub model input"""

#     size: int
#     tags: list[str]

#     @computed_field  # @field_validator("tasks", mode="after")
#     @property
#     def available_tasks(self) -> List[Tuple]:
#         """Filter hf tag tasks into edge coordinates"""
#         pattern = re.compile(r"(\w+)-to-(\w+)")
#         processed_tasks = []
#         data = NX_GRAPH.nodes.items()
#         conversion_pairs = dict(data).keys()
#         for each_tag in self.tags:
#             match = pattern.search(each_tag)
#             if match and match.group(1) in conversion_pairs and match.group(2) in conversion_pairs:
#                 processed_tasks.append((match.group(1), match.group(2)))
#         return processed_tasks


# class OllamaRegistryEntry(BaseModel):
#     """Validate ollama model input"""

#     size: int
#     tags: list[str]

#     @computed_field  # @field_validator("tasks", mode="after")
#     @property
#     def available_tasks(self) -> List[Tuple]:
#         """Filter ollama tag tasks into edge coordinates"""
#         processed_tasks = []
#         conversion_singlets = {"mllama": ("image", "text"), "llava": ("image", "text")}
#         for tasks in conversion_singlets:
#             if tasks in conversion_singlets:
#                 processed_tasks.append(conversion_singlets.get(tasks))
#             else:
#                 processed_tasks.append("text", "text")
#             return processed_tasks


# class ModelRegister(BaseModel):
#     """{model: {size: tasks}"""

#     model: Dict[int, list[str]]


# class GraphRegister(BaseModel):
#     model: Dict[int, list[tuple]]


# def from_ollama_cache() -> dict:
#     """Retrieve models from ollama server"""
#     response: ListResponse = ollama_list()
#     cache_dir = []
#     cache_sizes = []
#     model_tasks = []
#     for model in response.models:  # pylint: disable=no-member
#         cache_dir.append(f"ollama_chat/{model.model}")
#         cache_sizes.append(model.size.real)  # legible_size(
#         model_tasks.append([model.details.family])  # need a way to reduce this to smaller criteria
#     ollama_models = {model: {size: tasks} for model, size, tasks in zip(cache_dir, cache_sizes, model_tasks)}
#     return ollama_models


# def from_hf_hub() -> dict:
#     """Retrieve models from local huggingface hub cache"""
#     cached_repos = scan_cache_dir()
#     cache_dir = [obj.repo_id for obj in cached_repos.repos]
#     cache_sizes = [obj.size_on_disk for obj in cached_repos.repos]  # size_on_disk_str for human readable
#     metadata = []
#     for repo_name in cached_repos.repos:
#         metadata.append(repocard.RepoCard.load(repo_name.repo_id))
#     repo_details = [obj.data for obj in metadata]
#     model_tasks = []
#     for obj in repo_details:
#         current_tag = []
#         if hasattr(obj, "tags"):
#             current_tag.extend([*obj.tags])
#         if hasattr(obj, "pipeline_tag"):
#             current_tag.append(obj.pipeline_tag)
#         if current_tag is not None:
#             model_tasks.append(current_tag)  # need a way to reduce this to smaller criteria
#         else:
#             model_tasks.append("unknown")  # return to these and remove them from the list
#     hub_models = {model: {size: tasks} for model, size, tasks in zip(cache_dir, cache_sizes, model_tasks)}
#     return hub_models


# repo_size = "{:>12}".format(repo.size_on_disk_str),

# [speech] --edge --[text] --edge -- [image]

# tasks
# text-generation (text-to-text)
# text-to-image
# image-to-text (image to text)
# text-to-3d
# image-to-3d


# build graph
# print("  Family:", model.details.family) <--

# ollama reference -
# print("  Format:", model.details.format)
# print("  Parameter Size:", model.details.parameter_size)
# # print("  Quantization Level:", model.details.quantization_level)

# hf reference

# get repo data
# cached_repos = list(scan_cache_dir().repos)
# repo_name = next(iter(cached_repos)).repo_id
# repo_size = "{:>12}".format(repo.size_on_disk_str),
# str(repo.repo_path)
# repo.nb_files # idk

# for offline
# meta = repocard.RepoCard.load(repo_name)
# meta.data.tags (pipeline info)

# for online
# models = api.list_models(model_name=repo_name)
# type = next(iter(list(model_info))).pipeline_tag
# models = api.list_models(pipeline_tag="text-to-image", library="diffusers")

# separate keywords
# mllama (vllm), text-to-image, text-generation

import os

from nnll_01 import debug_monitor


@debug_monitor
def legible_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


@debug_monitor
def from_ollama_cache() -> dict:
    """Retrieve models from ollama server"""
    from ollama import ListResponse
    from ollama import list as ollama_list

    response: ListResponse = ollama_list()
    map_models = {}
    for model in response.models:  # pylint: disable=no-member
        if "/" in str(model.model):
            short_name = os.path.basename(str(model.model)).strip("[@]")
        else:
            short_name = str(model.model).strip("[@]")
        model_size_legible = legible_size(model.size.real)
        model_desc = f"▲ {short_name} - {model_size_legible} ▼"
        map_models.setdefault(model_desc, model.model)
    return map_models
