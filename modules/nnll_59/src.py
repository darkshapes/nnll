# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s
# # pylint: disable=line-too-long

import os
from typing import LiteralString

from huggingface_hub import hf_hub_download

from modules.nnll_60.src import CONFIG_PATH_NAMED, JSONCache


def download_hub_file(**kwargs) -> tuple:
    """
    Download a model from HuggingFace Hub and return path and blob names\n
    :param local_dir: `str` The local path to save the repo to
    :param repo_link: `str` The HF repository to download from
    :param file_name: `str` Name of the specific file to download
    :return: `tuple` the default download folder and folder contents
    """
    os.environ["HUGGINGFACE_HUB_CACHE"] = kwargs["local_dir"]
    hf_hub_download(**kwargs)
    if not kwargs["local_dir"]:
        repo_id = kwargs.get("repo_id")
        download_folder = "models--" + repo_id.replace("/", "--")
        folder_path_named = os.path.join(download_folder, "blobs")
        local_folder_path_named = os.path.join(folder_path_named)
    else:
        local_folder_path_named = kwargs["local_dir"]
    folder_contents = os.listdir(local_folder_path_named)
    return local_folder_path_named, folder_contents


config_data = JSONCache(CONFIG_PATH_NAMED)


@config_data.decorator
def name_save_file_as(extension: str = ".png", data=None) -> str:
    """
    Construct the file name of a save file\n
    :param extension: The extension of the file
    :type extension: `str`
    :param data: Auto-filled argument by decorator
    :type data: `None`
    :return: `str` A file path with a name
    """

    save_folder_path_absolute = data["save_folder_path_absolute"]
    if not os.path.isdir(save_folder_path_absolute):
        raise FileNotFoundError("Invalid folder location. {error_log}")
    files_in_save_location = os.listdir(save_folder_path_absolute)
    file_extension = extension
    file_count = sum(f.endswith(extension) for f in files_in_save_location)
    file_count = str(file_count).zfill(6)
    file_name = "Combo_" + file_count + file_extension
    file_path_absolute = os.path.join(save_folder_path_absolute, file_name)
    return file_path_absolute


def write_image_to_disk(image, metadata, extension: LiteralString = """.png"""):
    """Save image to file"""
    file_path_absolute = name_save_file_as(extension)
    image.save(file_path_absolute, "PNG", pnginfo=metadata)
    image.show()


def form_metadata(pipe, prompt, model_data, kwargs, negative_prompt=None):
    """Tabulate metadata"""
    gen_data = {
        "parameters": {
            "Prompt": prompt,
            "\nNegative prompt": negative_prompt if negative_prompt else "\n",
            "\nData": kwargs,
            "\nPipe": pipe,
            "\nModels": model_data,
        }
    }
    return gen_data
