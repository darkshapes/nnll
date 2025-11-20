# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

import os
from array import ArrayType
from pathlib import Path
from typing import Any

import PIL.Image

from nnll import HOME_FOLDER_PATH, USER_PATH_NAMED
from nnll.read_tags import MetadataFileReader
from nnll.constants import ExtensionType


def name_save_file_as(extension: ExtensionType) -> Path:
    """Construct the file name of a save file\n
     :param extension: The extension of the file
    b :return: `str` A file path with a name"""

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
def write_to_disk(content: Any, metadata: dict[str], extension: str = None, **kwargs) -> None:
    """Save Image to File\n
    :param content: File data in memory
    :param pipe_data: Pipe metadata to write into the file
    ```
        name    [ header:  type   : medium: type ]

                ,-pipe      dict
            #   \-model     str    ,-text   string
            #   \-prompt    dict___\-audio  array
            #   `-kwargs    dict   \-image  array
            #                      `-video  array
    ```
    :param extension: Type of file to write, defaults to None
    :param library: Originating library, defaults to None\n\n"""

    file_path_absolute = name_save_file_as(next(iter(extension)))
    if extension == ExtensionType.GIF_:
        from diffusers.utils import export_to_gif

        export_to_gif(content, file_path_absolute)
    elif isinstance(content, PIL.Image.Image):
        from PIL import PngImagePlugin

        embed = PngImagePlugin.PngInfo()
        embed.add_text("parameters", str(metadata))
        file_suffix = extension.pop()
        file_suffix.strip(".")
        kwargs.setdefault("pnginfo", embed)
        content.save(file_path_absolute, file_suffix.upper().strip("."), **kwargs)
        content.show()
    elif extension == ExtensionType.PNG_:  # MFLUX case
        file_suffix = extension.pop()
        file_suffix.strip(".")
        content.save(file_path_absolute, file_suffix.upper().strip("."))

    elif isinstance(content, ArrayType) and extension in ExtensionType.AUDIO:
        if kwargs.get("library") == ["audiocraft"]:
            from audiocraft.data.audio import audio_write  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            for idx, one_wav in enumerate(content):
                audio_write(f"{file_path_absolute}{idx}", one_wav.cpu(), metadata, strategy="loudness", loudness_compressor=True)
        elif kwargs.get("library") == "scipy":
            import scipy

            scipy.io.wavfile.write(file_path_absolute, rate=16000, data=content)
        else:
            import soundfile as sf  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            kwargs.pop("library", "")
            sf.write(file_path_absolute, content, kwargs.get("sampling_rate"))
    elif extension in ExtensionType.VIDEO:
        # `imageio` / `imageio-ffmpeg` ??
        from diffusers.utils import export_to_video

        export_to_video(content, file_path_absolute, fps=15)
