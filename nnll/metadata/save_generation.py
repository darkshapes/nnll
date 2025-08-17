# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

import os
from array import ArrayType
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import PIL.Image

from nnll.configure import HOME_FOLDER_PATH, USER_PATH_NAMED
from nnll.metadata.read_tags import MetadataFileReader
from nnll.monitor.file import debug_monitor


@debug_monitor
def name_save_file_as(extension: Literal[".png", ".wav", ".jpg"] = ".png") -> Path:
    """Construct the file name of a save file\n
    :param extension: The extension of the file
    :return: `str` A file path with a name"""

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
def write_to_disk(content: Any, metadata: dict[str], extension: str = None, library: Optional[str] = None, output_type: str | None = None) -> None:
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

    file_path_absolute = name_save_file_as(extension)
    if isinstance(content, PIL.Image.Image):
        from PIL import PngImagePlugin

        embed = PngImagePlugin.PngInfo()
        embed.add_text("parameters", str(metadata))
        content.save(file_path_absolute, extension.upper().strip("."), pnginfo=embed)
        content.show()

    elif extension.strip(".") == "gif":
        from diffusers.utils import export_to_gif

        export_to_gif(content, file_path_absolute)
    elif isinstance(content, ArrayType):
        if library == ["audiocraft"]:
            from audiocraft.data.audio import audio_write  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            for idx, one_wav in enumerate(content):
                audio_write(f"{file_path_absolute}{idx}", one_wav.cpu(), metadata, strategy="loudness", loudness_compressor=True)
        elif output_type == "scipy":
            import scipy

            scipy.io.wavfile.write(file_path_absolute, rate=16000, data=content)
        else:
            import soundfile as sf  # pyright: ignore[reportMissingImports] | pylint:disable=import-error

            sf.write(file_path_absolute, content, metadata)
    else:
        # `imageio` / `imageio-ffmpeg` ??
        from diffusers.utils import export_to_video

        export_to_video(content, file_path_absolute, fps=15)
