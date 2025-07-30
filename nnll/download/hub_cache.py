# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import os
from functools import wraps
from typing import Callable, Dict, Tuple


def manage_env(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        source = kwargs.get("source", "huggingface")
        local_dir = kwargs.get("local_dir", None)

        env_var = f"{source.upper()}_HUB_CACHE" if source != "modelscope" else "MODELSCOPE_CACHE"
        revert = os.environ.get(env_var)

        try:
            if kwargs.get("local_dir", 0):
                os.environ[env_var] = local_dir
            return func(*args, **kwargs)
        finally:
            if local_dir and revert is not None:
                os.environ[env_var] = revert

    return wrapper


@manage_env
def download_hub_file(repo_id: str, source: str = "huggingface", **kwargs) -> Tuple[str, list]:
    """
    Download a model from various hub sources and return path and blob names.\n
    :param local_dir: The local path to save the repo to
    :param repo_id: The repository ID to download from
    :param file_name: Name of the specific file to download
    :param source: Remote hub host to use ("huggingface", "kaggle", "modelscope")
    :return: A tuple containing the default download folder and folder contents
    """

    from nnll.monitor.file import dbuq

    if source == "huggingface":
        from huggingface_hub import hf_hub_download, constants  # pylint:disable=import-error

        constants.HF_HUB_OFFLINE = 0
        os.environ["HF_HUB_OFFLINE"] = "0"

        downloader = hf_hub_download
    elif source == "kaggle":
        import kagglehub  # pylint:disable=import-error

        downloader = kagglehub.model_download
    elif source == "modelscope":
        import modelscope  # pylint:disable=import-error

        downloader = modelscope.hub.snapshot_download

    download_functions: Dict[str, Callable] = {
        "huggingface": lambda **kwargs: downloader(repo_id=repo_id, **kwargs),
        "kaggle": lambda **kwargs: downloader(repo_id, **kwargs),
        "modelscope": lambda **kwargs: downloader(repo_id, cache_dir=kwargs.get("local_dir"), **kwargs),
    }

    if source not in download_functions:
        raise ValueError(f"Unsupported source: {source}")

    try:
        download_functions[source](**kwargs)
    except TimeoutError as error_log:
        if source == "huggingface":
            constants.HF_HUB_OFFLINE = 1
            os.environ["HF_HUB_OFFLINE"] = "1"
        dbuq(error_log)
    if not kwargs.get("local_dir"):
        repo_id = kwargs.get("repo_id")
        download_folder = "models--" + repo_id.replace("/", "--")
        local_folder_path_named = os.path.join(download_folder, "blobs")
    else:
        local_folder_path_named = kwargs["local_dir"]
    folder_contents = os.listdir(local_folder_path_named)  # Did not use 'Path' type b/c might be relative directory??
    return local_folder_path_named, folder_contents


async def get_hub_path(repo_id: str) -> str:
    """Returns the file path from a repository based on a query.\n
    :param repo: Repository object with revisions and files information.
    :return: The matched file path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_files_only=True)


async def get_hub_layers(repo_id: str) -> dict[str, dict[str, str | list[int] | None]]:
    """Map the inner layers of a Safetensors model state dict from the hub\n
    dtype, shape, data_offsets
    :param repo_id: Hub location to index
    :return: The model layer metadata"""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import NotASafetensorsRepoError
    from urllib3.exceptions import NameResolutionError, MaxRetryError
    from requests.exceptions import ConnectionError
    from socket import gaierror
    import json

    os.environ["HF_HUB_OFFLINE"] = "0"

    hf_api = HfApi()
    try:
        api_data = hf_api.get_safetensors_metadata(repo_id).files_metadata
    except (NotASafetensorsRepoError, NameResolutionError, MaxRetryError, ConnectionError, gaierror):
        return None
    else:
        metadata = dict()
        for safetensor, tensor_info in api_data.items():
            tensor_data = tensor_info.tensors
            for layer_name, layer_data in tensor_data.items():
                data_bundle = {"dtype": layer_data.dtype, "shape": layer_data.shape, "data_offsets": list(layer_data.data_offsets)}
                metadata.setdefault(layer_name, data_bundle)
    metadata = json.dumps(metadata, sort_keys=True)
    return metadata
