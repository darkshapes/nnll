### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

from array import ArrayType
import os
from typing import Literal, Any, Optional, Dict
from pathlib import Path
import PIL.Image
from nnll.monitor.file import debug_monitor
from nnll.metadata.read_tags import MetadataFileReader
from nnll.integrity.hashing import collect_hashes
from nnll.configure import USER_PATH_NAMED, HOME_FOLDER_PATH


@debug_monitor
def name_save_file_as(extension: Literal[".png", ".wav", ".jpg"] = ".png") -> Path:
    """
    Construct the file name of a save file\n
    :param extension: The extension of the file
    :return: `str` A file path with a name
    """
    file_reader = MetadataFileReader()
    user_settings = file_reader.read_header(USER_PATH_NAMED)
    save_folder_path_absolute = user_settings["location"].get("output", os.getcwd())  # pylint: disable=unsubscriptable-object
    if save_folder_path_absolute == "output":
        save_folder_path_absolute = os.path.join(HOME_FOLDER_PATH, "output")
    if not os.path.isdir(save_folder_path_absolute):
        raise FileNotFoundError("Invalid folder location. {error_log}")
    files_in_save_location = os.listdir(save_folder_path_absolute)
    file_extension = extension
    file_count = sum(f.endswith(extension) for f in files_in_save_location)
    file_count = str(file_count).zfill(6)
    file_name = "Shadowbox_" + file_count + file_extension
    file_path_absolute = os.path.join(save_folder_path_absolute, file_name)
    return file_path_absolute


# do not log here
def add_to_metadata(pipe: Dict, model: str, prompt: str | list[str] | dict[str], kwargs: dict) -> Dict:  #  negative_prompt=None
    """
    Create metadata from active hf inference pipes\n
    :param pipe: Active HuggingFace pipe from diffusers/transformers
    :param model: Generative model filename/path
    :param prompt: Input for genereation
    :param kwargs: Arguments passed to constructors
    :return: Dictionary of attributes
    """
    model_data = {}
    model_data.setdefault(model, collect_hashes(model))  # adapt for cache repo locations

    gen_data = {
        "parameters": {
            "Prompt": prompt,
            # "\nNegative prompt": negative_prompt if negative_prompt else "\n",
            "\nData": kwargs,
            "\nPipe": pipe,
            "\nModels": model_data,
        }
    }
    return gen_data


# do not log here
def write_to_disk(content: Any, metadata: dict[str], extension: str = None, library: Optional[str] = None) -> None:
    """Save Image to File\n
    :param image: Image file data in memory
    :param metadata: Pipe metadata to write into the file
    :param extension: Type of Image file to write, defaults to
    """
    if isinstance(content, PIL.Image.Image):
        from PIL import PngImagePlugin

        file_path_absolute = name_save_file_as(".png")
        embed = PngImagePlugin.PngInfo()
        embed.add_text("parameters", str(metadata))
        content.save(file_path_absolute, "PNG", pnginfo=embed)
        content.show()

    elif isinstance(content, ArrayType):
        if library == ["audiocraft"]:
            from audiocraft.data.audio import audio_write  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            for idx, one_wav in enumerate(content):
                audio_write(f"{name_save_file_as('.wav')}{idx}", one_wav.cpu(), metadata, strategy="loudness", loudness_compressor=True)
        else:
            import soundfile as sf  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            file_path_absolute = name_save_file_as(extension or ".wav")
            sf.write(file_path_absolute, content, metadata)
