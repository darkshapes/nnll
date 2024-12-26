
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
from pathlib import Path
import itertools

from modules.nnll_04.src import metadata_from_safetensors
from modules.nnll_05.src import metadata_from_gguf
from modules.nnll_28.src import load_pickletensor_metadata_from_model

def get_model_header(target_file: str) -> tuple:
    """
    Detect file type and skim metadata from a model file using the appropriate tools\n
    :param target_file: `str` The full path to the file being analyzed
    :return: `tuple` Four values constituting the metadata header and external file attributes\n
    (model_header, disk_size, file_name, file_extension)
    """
    safetensors_loader = None
    gguf_loader = None
    pickletensor_loader = None

    #move this and import statements to .json file
    safetensors_extensions = [
        ".safetensors",
        ".sft"
        ]

    gguf_extensions =  [".gguf"]

    pickletensor_extensions = [
        ".pt",
        ".pth",
        ".ckpt",
    ]

    extensions_list = itertools.chain(safetensors_extensions, gguf_extensions, pickletensor_extensions)
    # Get external file metadata
    file_extension = Path(target_file).suffix.lower()
    if file_extension == '' or file_extension is None or file_extension not in list(extensions_list):  # Skip file if we cannot possibly know what it is
        return
    else:
        if file_extension in safetensors_extensions:
            open_header_method = metadata_from_safetensors
        elif file_extension in gguf_extensions:
            open_header_method = metadata_from_gguf
        elif file_extension in pickletensor_extensions:
            open_header_method = load_pickletensor_metadata_from_model

        file_name = os.path.basename(target_file)
        disk_size = os.path.getsize(target_file)

        # Retrieve header by method indicated by extension, usually struct unpacking, except for pt files which are memmap
        model_header = open_header_method(target_file)
        return (model_header, disk_size, file_name, file_extension)
