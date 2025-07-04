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
