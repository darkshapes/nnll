#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import ollama
# import huggingface_hub

from huggingface_hub import scan_cache_dir


def from_ollama_cache() -> dict:
    """Retrieve models from ollama server"""
    available_models = {}
    response: ollama.ListResponse = ollama.list()
    for model in response.models:  # pylint: disable=no-member
        available_models.setdefault(f"{model.model}-{(model.size.real / 1024 / 1024):.2f} MB", model.model)
    return available_models


# consider export HF_HUB_OFFLINE=True
# export DISABLE_TELEMETRY=YES
# set DISABLE_TELEMETRY=YES
# HF_HOME
# HUGGINFACE_HUB_CACHE


def from_hf_cache_() -> dict:
    cached_repos = list(scan_cache_dir().repos)

    #     repo.repo_id,
    # repo.repo_type,
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
