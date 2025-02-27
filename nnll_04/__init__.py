### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import struct
import json


def metadata_from_safetensors(file_path_named: str) -> dict:
    """
    Collect metadata from a safetensors file header\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """
    with open(file_path_named, "rb") as file_contents_to:
        first_8_bytes = file_contents_to.read(8)
        length_of_header = struct.unpack("<Q", first_8_bytes)[0]
        header_content_bytes = file_contents_to.read(length_of_header)
        header_contents = json.loads(header_content_bytes.decode("utf-8"))
        # we want to remove this metadata so its not counted as tensors
        if header_contents.get("__metadata__", 0) != 0:
            # it is usually empty on safetensors ._.
            header_contents.pop("__metadata__")
        return header_contents
