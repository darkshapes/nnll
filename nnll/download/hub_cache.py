# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

import os
from functools import wraps
from typing import Callable, Dict, Tuple
from nnll.monitor.console import nfo


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

    from pathlib import Path

    from nnll.monitor.file import dbuq

    called_with_filename = kwargs.get("filename")
    if source == "huggingface":
        if called_with_filename:
            from huggingface_hub import constants
            from huggingface_hub import hf_hub_download as downloader  # pylint:disable=import-error
        else:
            from huggingface_hub import constants
            from huggingface_hub import snapshot_download as downloader  # pylint:disable=import-error

            kwargs.setdefault("repo_type", "model")
            kwargs.setdefault("allow_patterns", ["*.safetensors*", "*.gguf", "*.onnx", "*.pth"])

        constants.HF_HUB_OFFLINE = 0
        os.environ["HF_HUB_OFFLINE"] = "0"

    elif source == "kaggle":
        from kagglehub import model_download as downloader  # pylint:disable=import-error

    elif source == "modelscope":
        from modelscope import snapshot_download as downloader  # pylint:disable=import-error

    download_functions: Dict[str, Callable] = {
        "huggingface": lambda **kwargs: downloader(repo_id=repo_id, **kwargs),
        "kaggle": lambda **kwargs: downloader(repo_id, **kwargs),
        "modelscope": lambda **kwargs: downloader(repo_id, cache_dir=kwargs.get("local_dir"), **kwargs),
    }

    if source not in download_functions:
        raise ValueError(f"Unsupported source: {source}")

    try:
        if called_with_filename:
            download_functions[source](**kwargs)
        else:
            download_functions[source](**kwargs)

    except TimeoutError as error_log:
        if source == "huggingface":
            constants.HF_HUB_OFFLINE = 1
            os.environ["HF_HUB_OFFLINE"] = "1"
        dbuq(error_log)
    if not kwargs.get("local_dir"):
        download_folder = "models--" + repo_id.replace("/", "--")
        if source == "huggingface" and not called_with_filename:
            local_folder_path_named = os.path.join(Path.home(), ".cache", "huggingface", "hub", download_folder, "snapshots")
        elif source == "huggingface" and called_with_filename:
            download_folder = os.path(constants.HF_HUB_CACHE, download_folder)
            local_folder_path_named = os.path.join(download_folder, "blobs")
    else:
        local_folder_path_named = kwargs["local_dir"]
    if source == "huggingface":
        constants.HF_HUB_OFFLINE = 1
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not os.path.exists(local_folder_path_named):
        try:
            raise TypeError(f"{local_folder_path_named} does not exist")
        except TypeError as error_log:
            nfo(error_log)
            return

    folder_contents = os.listdir(local_folder_path_named)  # Did not use 'Path' type b/c might be relative directory??
    return local_folder_path_named, folder_contents


async def get_hub_path(repo_id: str) -> str:
    """Returns the file path from a repository based on a query.\n
    :param repo: Repository object with revisions and files information.
    :return: The matched file path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_files_only=True)


async def get_hub_layers(file_or_repo_path: str, local: bool = False) -> dict[str, dict[str, str | list[int] | None]]:
    """Map the inner layers of a Safetensors model state dict from the cache\n
    dtype, shape, data_offsets
    :param repo_id: Hub location to index
    :return: The model layer metadata"""

    import json
    from socket import gaierror

    from huggingface_hub.errors import NotASafetensorsRepoError
    from requests.exceptions import ConnectionError
    from urllib3.exceptions import MaxRetryError, NameResolutionError

    if local:
        from huggingface_hub import load_state_dict_from_file as state_dict_loader

        loader_kwargs = {"checkpoint_file": file_or_repo_path}

        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        state_dict_loader = hf_api.get_safetensors_metadata
        loader_kwargs = {"repo_id": file_or_repo_path}
        os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        api_data = state_dict_loader(**loader_kwargs)
    except (NotASafetensorsRepoError, NameResolutionError, MaxRetryError, ConnectionError, gaierror):
        from nnll.monitor.console import nfo

        nfo("Couldn't find safetensors format")
        return None
    else:
        metadata = dict()
        api_data
        for safetensor, tensor_info in api_data.items():
            tensor_data = tensor_info.tensors
            for layer_name, layer_data in tensor_data.items():
                data_bundle = {"dtype": layer_data.dtype, "shape": layer_data.shape, "data_offsets": list(layer_data.data_offsets)}
                metadata.setdefault(layer_name, data_bundle)
    metadata = json.dumps(metadata, sort_keys=True)
    return metadata


# def mass_download(local_folder: str, repo_id_stack: List):
#     from huggingface_hub import snapshot_download

#     for repo_id in repo_id_stack:
#         snapshot_download(local_dir=local_folder, repo_id=repo_id, allow_patterns=["*.safetensors*", "*.gguf","*.onnx"])


def main() -> None:
    from nnll.monitor.console import nfo
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="""Generate hash files for a remote or cached repo""")
    parser.add_argument("repo", type=str, help="Relative path to repository")
    args = parser.parse_args()
    repo_id = args.repo

    nfo(repo_id)
    contents = download_hub_file(repo_id=repo_id, source="huggingface")

    hash_kwargs = [
        {"layer": False, "b3": False},
        {"layer": True, "b3": False},
        {"layer": True, "b3": True},
    ]
    from nnll.integrity.hash_256 import hash_layers_or_files, write_to_file

    if contents:
        nfo(contents)
        for kwarg_set in hash_kwargs:
            hashes = {}
            for root, folders, files in os.walk(contents[0]):
                hashed_layers = asyncio.run(hash_layers_or_files(folder_path=root, **kwarg_set))
                for file_path in hashed_layers:
                    hashes.setdefault(file_path, hashed_layers[file_path])
            file_name_prefix = "b3_" if kwarg_set["b3"] else "sha256_"
            file_name_suffix = (f"layer_{os.path.basename(repo_id)}" if kwarg_set["layer"] else f"file_{os.path.basename(repo_id)}",)
            asyncio.run(
                write_to_file(
                    program=f"{file_name_prefix}{file_name_suffix}",
                    folder_path_named=root,
                    hash_values=hashes,
                    write_path=".",
                )
            )

            nfo(hashes)


if __name__ == "__main__":
    main()
