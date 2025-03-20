### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import LiteralString
import os

save_folder_name_and_path = os.path.dirname(__file__)


def save_element(extension=".png", save_folder_name_and_path=save_folder_name_and_path):
    from PIL import Image

    print(save_folder_name_and_path)
    if not os.path.isdir(save_folder_name_and_path):
        raise FileNotFoundError("Invalid folder location.")
    else:
        files_in_save_location = os.listdir(save_folder_name_and_path)
        file_extension = extension
        file_count = sum(f.endswith(file_extension) for f in files_in_save_location)
        file_count = str(file_count).zfill(6)
        file_name = "Combo_" + file_count + file_extension
        file_path_absolute = os.path.join(save_folder_name_and_path, file_name)
        return file_path_absolute


def write_to_disk(image, metadata, extension: LiteralString = """.png"""):
    file_path_absolute = save_element(extension)
    img = image.save(file_path_absolute, "PNG", pnginfo=metadata)
    img.show()
