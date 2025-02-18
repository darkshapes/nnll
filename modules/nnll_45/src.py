# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import os


def download_hub_file(**kwargs) -> tuple:
    """
    Download a model from HuggingFace Hub and return path and blob names\n
    :param local_dir: `str` The local path to save the repo to
    :param repo_link: `str` The HF repository to download from
    :param file_name: `str` Name of the specific file to download
    :return: `tuple` the default download folder and folder contents
    """
    try:
        os.environ["HUGGINGFACE_HUB_CACHE"] = kwargs["local_dir"]
        from huggingface_hub import hf_hub_download

        hf_hub_download(**kwargs)
    except ImportError as error_log:
        print(f"{error_log} huggingface_hub not installed.")
    else:
        if not kwargs["local_dir"]:
            repo_id = kwargs.get("repo_id")
            download_folder = "models--" + repo_id.replace("/", "--")
            folder_path_named = os.path.join(download_folder, "blobs")
            local_folder_path_named = os.path.join(folder_path_named)
        else:
            local_folder_path_named = kwargs["local_dir"]
        folder_contents = os.listdir(local_folder_path_named)
        return local_folder_path_named, folder_contents
