
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import os
from collections import defaultdict
import struct

GGUF_MAGIC_NUMBER = b"GGUF"

def gguf_check(file_path_named: str) ->tuple:
    """
    A magic word check to ensure a file is GGUF format\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `tuple' the number
    """
    try:
        with open(file_path_named, "rb") as file_contents_to:
            magic_number = file_contents_to.read(4)
            version = struct.unpack("<I", file_contents_to.read(4))[0]
    except (ValueError, Exception) as e:
        print(f"Error reading GGUF header from {file_path_named}: {e}")
        return None
    else:
        if not magic_number and magic_number!= GGUF_MAGIC_NUMBER:
            print(f"Invalid GGUF magic number in '{file_path_named}'")
            return False
        elif version < 2:
            print(f"Unsupported GGUF version {version} in '{file_path_named}'")
            return False
        elif magic_number == GGUF_MAGIC_NUMBER and version >= 2:
            return True
        else:
            return False


def create_llama_parser(file_path_named: str) -> dict:
    """
    Llama handler for gguf file header\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `dict` The entire header with Llama parser formatting
    """
    try:
        from llama_cpp import Llama
    except ImportError as error_log:
        ImportError(f"{error_log} llama_cpp not installed.")
    else:
        parser = Llama(model_path=file_path_named, vocab_only=True, verbose=False)
        return parser


def metadata_from_gguf(file_path_named: str) -> dict:
    """
    Collect metadata from a gguf file header\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """

    if gguf_check(file_path_named) != True:
        return

    else:
        parser = create_llama_parser(file_path_named)
        if parser:
            file_metadata = defaultdict(dict)
            file_metadata["name"] = next(
                (value for key in ["general.basename", "general.base_model.0", "general.name", "general.architecture"]
                if (value := parser.metadata.get(key)) is not None),
                None
            )
            file_metadata["dtype"] = getattr(parser.scores, 'dtype', None) and parser.scores.dtype.name  # e.g., 'float32'
            return file_metadata
