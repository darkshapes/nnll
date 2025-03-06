#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
import ollama
# import huggingface_hub

from huggingface_hub import scan_cache_dir


def legible_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def from_ollama_cache() -> dict:
    """Retrieve models from ollama server"""
    response: ollama.ListResponse = ollama.list()
    map_models = {}
    for model in response.models:  # pylint: disable=no-member
        if "/" in str(model.model):
            short_name = os.path.basename(str(model.model))
        else:
            short_name = str(model.model)
        map_models.setdefault(f"{short_name} - {legible_size(model.size.real)}  ⭥", f"ollama_chat/{model.model}")
        # available_models.append(tuple([f"{model.model} - {(model.size.real / 1024 / 1024):.2f} MB", f"ollama_chat/{model.model}"]))
    return map_models

    # consider export HF_HUB_OFFLINE=True
    # export DISABLE_TELEMETRY=YES
    # set DISABLE_TELEMETRY=YES
    # HF_HOME
    # HUGGINFACE_HUB_CACHE

    # def from_hf_cache_() -> dict:
    #     cached_repos = list(scan_cache_dir().repos)

    #     #     repo.repo_id,
    #     # repo.repo_type,
    # "{:>12}".format(repo.size_on_disk_str),
    # repo.nb_files,
    # repo.last_accessed_str,
    # repo.last_modified_str,
    # str(repo.repo_path),
    """Retrieve models from huggingface hub cache server"""
    available_models = {}
    # response: ollama.ListResponse = ollama.list()
    # for model in response.models:
    #     available_models.setdefault(f"{model.model}-{(model.size.real / 1024 / 1024):.2f} MB", model.model)
    # return available_models
    # if model.details:
    #     print("  Format:", model.details.format)
    #     print("  Family:", model.details.family)
    #     print("  Parameter Size:", model.details.parameter_size)
    #     print("  Quantization Level:", model.details.quantization_level)
    # print("\n")
