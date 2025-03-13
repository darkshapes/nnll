#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import os
import ollama


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
            short_name = os.path.basename(str(model.model)).strip("[@]")
        else:
            short_name = str(model.model).strip("[@]")
        model_size_legible = legible_size(model.size.real)
        model_desc = f"{short_name} - {model_size_legible}"
        map_models.setdefault(model_desc, model.model)
    return map_models

    # build graph
    # ollama reference -

    # print("  Family:", model.details.family) <--
    # print("  Parameter Size:", model.details.parameter_size)
    # # print("  Quantization Level:", model.details.quantization_level)

    # hf reference
    # from huggingface_hub import scan_cache_dir

    # cached_repos = list(scan_cache_dir().repos)
    # repo_name = next(iter(cached_repos)).repo_id
    # meta = repocard.RepoCard.load(repo_name)
    # meta.data.tags (pipeline info)

    # models = api.list_models(model_name=repo_name)
    # type = next(iter(list(model_info))).pipeline_tag

    # models = api.list_models(pipeline_tag="text-to-image", library="diffusers")

    # encoding = tiktoken.get_encoding("cl100k_base")
    # token_count = len(encoding.encode(message))

    #  # import huggingface_hub

    # from huggingface_hub import scan_cache_dir

    # def from_hf_cache_() -> dict:
    #     cached_repos = list(scan_cache_dir().repos)

    #     #     repo.repo_id,
    #     # repo.repo_type,
    # "{:>12}".format(repo.size_on_disk_str),
    # repo.nb_files,
    # repo.last_accessed_str,
    # repo.last_modified_str,
    # str(repo.repo_path),
    # """Retrieve models from huggingface hub cache server"""
    # available_models = {}
    # response: ollama.ListResponse = ollama.list()
    # for model in response.models:
    #     available_models.setdefault(f"{model.model}-{(model.size.real / 1024 / 1024):.2f} MB", model.model)
    # return available_models
    # if model.details:
    #     print("  Format:", model.details.format)

    # print("\n")
