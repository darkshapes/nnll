### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

from io import TextIOWrapper
import os
from typing import LiteralString, Dict, List

import PIL
from nnll_01 import debug_monitor
from nnll_44 import collect_hashes
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

config_data = JSONCache(CONFIG_PATH_NAMED)


@debug_monitor
@config_data.decorator
async def name_save_file_as(extension: str = ".png", data: TextIOWrapper = None) -> str:
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


@debug_monitor
async def add_to_metadata(pipe: Dict, model: str, prompt: str | list[str] | dict[str], kwargs: dict) -> Dict:
    """
    Create metadata from active hf inference pipes\n
    :param pipe: Active HuggingFace pipe from diffusers/transformers
    :param model: Generative model filename/path
    :param prompt: Input for genereation
    :param kwargs: Arguments passed to constructors
    :return: Dictionary of attributes
    """
    model_data = {}
    model_data.setdefault(collect_hashes(model))

    gen_data = {
        "parameters": {
            "Prompt": prompt,
            "\nData": kwargs,
            "\nPipe": pipe,
            "\nModels": model_data,
        }
    }
    return gen_data


@debug_monitor
async def write_image_to_disk(image: PIL, metadata: dict[str], extension: LiteralString = """.png"""):
    """Save image to file"""
    file_path_absolute = name_save_file_as(extension)
    image.save(file_path_absolute, "PNG", pnginfo=metadata)
    image.show()
