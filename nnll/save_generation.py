# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

import os
from einops import rearrange
from pathlib import Path
import torch


from nnll.constants import ExtensionType
from nnll.console import nfo
from nnll.hyperchain import HyperChain
from nnll.helpers import ensure_path


def name_save_file_as(extension: set[str], save_folder_path=".output") -> Path:
    """Construct the file name of a save file\n
     :param extension: The extension of the file
    b :return: `str` A file path with a name"""

    if not os.path.isdir(save_folder_path):
        try:
            ensure_path(save_folder_path)
        except Exception:
            nfo("Invalid folder location. {error_log}")
    files_in_save_location = os.listdir(save_folder_path)
    file_extension = next(iter(extension))
    file_count = sum(f.endswith(file_extension) for f in files_in_save_location)
    file_count = str(file_count).zfill(6)
    file_name = "divisor_" + file_count + file_extension
    file_path_named = os.path.join(save_folder_path, file_name)
    return file_path_named


def save_with_hyperchain(
    file_path_named: str | Path,
    tensor: torch.Tensor,
    hyperchain: HyperChain,
    extension: set[str],
) -> None:
    nfo(f"Saving {file_path_named}")
    if extension == ExtensionType.WEBP:
        from PIL import Image
        from PIL.ExifTags import Base

        latent = tensor.clamp(-1, 1)
        numeric = rearrange(latent[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (numeric + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        hyperchain_metadata = str(hyperchain)

        # Store hyperchain data in ImageDescription (EXIF tag 270)
        exif_data[Base.ImageDescription] = hyperchain_metadata
        exif_data[Base.Software] = "divisor"

        img.save(file_path_named, format="WEBP", lossless=True, exif=exif_data)
