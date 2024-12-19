
import os
from collections import defaultdict
import struct
from llama_cpp import Llama


def read_gguf_header(file_path: str):
    try:
        with open(file_path, "rb") as file:
            magic = file.read(4)
            version = struct.unpack("<I", file.read(4))[0]
            return magic, version
    except (ValueError, Exception) as e:
        print(f"Error reading GGUF header from {file_path}: {e}")
        return None


def parse_gguf_model(file_path: str):
    parser = Llama(model_path=file_path, vocab_only=True, verbose=False)
    return parser


def load_gguf_metadata(file_path: str) -> dict:
    """
    Collect metadata from a gguf file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """

    file_metadata = defaultdict(dict)

    # File size for memory management
    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return {}

    file_metadata["size"] = file_size

    header_info = read_gguf_header(file_path)
    if header_info is None or header_info[0] != b"GGUF":
        print(f"Invalid GGUF magic number in '{file_path}'")
        return {}

    magic, version = header_info
    if version < 2:
        print(f"Unsupported GGUF version {version} in '{file_path}'")
        return {}

    parser = parse_gguf_model(file_path)
    if not parser:
        return file_metadata

    arch = parser.metadata.get("general.architecture", "")
    name = parser.metadata.get("general.name", "")
    file_metadata["name"] = name or arch

    file_metadata["dtype"] = getattr(parser.scores, 'dtype', None) and parser.scores.dtype.name  # e.g., 'float32'

    return file_metadata
