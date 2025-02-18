# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import os
import modules.nnll_57.look as look
from PIL import Image
from typing import LiteralString


def save_element(extension=".png", save_folder_name_and_path=look.save_folder_name_and_path):
    print(save_folder_name_and_path)
    if not os.path.isdir(save_folder_name_and_path):
        raise FileNotFoundError("Invalid folder location. {error_log}")
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
