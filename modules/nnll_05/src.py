
import os
from collections import defaultdict
import struct


def read_gguf_header(file_path: str) ->tuple:
    """
    A magic word check to ensure a file is GGUF format\n
    :param file_path: `str` the full path to the file being opened
    :return: `tuple' the number
    """
    try:
        with open(file_path, "rb") as file:
            magic = file.read(4)
            version = struct.unpack("<I", file.read(4))[0]
    except (ValueError, Exception) as e:
        print(f"Error reading GGUF header from {file_path}: {e}")
        return None
    else:
        if magic is None or magic != b"GGUF":
            print(f"Invalid GGUF magic number in '{file_path}'")
            return False
        elif version < 2:
            print(f"Unsupported GGUF version {version} in '{file_path}'")
            return False
        elif magic == b"GGUF" and version >= 2:
            return True
        else:
            return False



def parse_gguf_model(file_path: str) -> dict:
    """
    Llama handler for gguf file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` The entire header with Llama parser formatting
    """
    try:
        from llama_cpp import Llama
    except ImportError as error_log:
        ImportError(f"{error_log} llama_cpp not installed.")
    else:
        parser = Llama(model_path=file_path, vocab_only=True, verbose=False)
        return parser


def load_gguf_metadata(file_path: str) -> dict:
    """
    Collect metadata from a gguf file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """

    if read_gguf_header(file_path) != True:
        return

    else:
        parser = parse_gguf_model(file_path)
        if parser:
            file_metadata = defaultdict(dict)
            arch = parser.metadata.get("general.architecture", "")
            name = parser.metadata.get("general.name", "")
            file_metadata["name"] = name or arch
            file_metadata["dtype"] = getattr(parser.scores, 'dtype', None) and parser.scores.dtype.name  # e.g., 'float32'
            print(file_metadata)
            return file_metadata
