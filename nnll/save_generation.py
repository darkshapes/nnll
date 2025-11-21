# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

import os
from array import ArrayType
from pathlib import Path
from typing import Any
import PIL.Image

from nnll.constants import ExtensionType


def name_save_file_as(extension: ExtensionType, save_folder_path=".output") -> Path:
    """Construct the file name of a save file\n
     :param extension: The extension of the file
    b :return: `str` A file path with a name"""

    if not os.path.isdir(save_folder_path):
        raise FileNotFoundError("Invalid folder location. {error_log}")
    files_in_save_location = os.listdir(save_folder_path)
    file_extension = extension
    file_count = sum(f.endswith(extension) for f in files_in_save_location)
    file_count = str(file_count).zfill(6)
    file_name = "divisor_" + file_count + file_extension
    file_path_named = os.path.join(save_folder_path, file_name)
    return file_path_named


# do not log here
def write_to_disk(content: Any, metadata: dict[str], extension: ExtensionType | None = None, **kwargs) -> None:
    """Save Image to File\n
    :param content: File data in memory
    :param pipe_data: Pipe metadata to write into the file
    ```
    #   name    [ header:  type   : medium: type ]

            #    ,-pipe      dict
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
    if extension == ExtensionType.WEBP:
        import numpy as np
        from PIL import Image

        if not file_path_absolute.endswith(ExtensionType.WEBP):
            filename = file_path_absolute.rsplit(".", 1)[0] + ExtensionType.WEBP

        # Convert tensor to PIL Image
        if content.dim() == 4:
            image = content[0]  # Take first batch item
        if image.dim() == 3:
            image_tensor = image.clamp(0, 1).permute(1, 2, 0).cpu()
            image_np: np.ndarray = image_tensor.numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")

        pil_image.save(filename, "WEBP", lossless=True)
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
