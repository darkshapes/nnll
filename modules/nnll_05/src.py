
import os
from collections import defaultdict
import struct
from llama_cpp import Llama
from pathlib import Path


def read_gguf_header(file_name: str):
    try:
        with open(file_name, "rb") as file:
            magic = file.read(4)
            version = struct.unpack("<I", file.read(4))[0]
            return magic, version
    except (ValueError, Exception) as e:
        print(f"Error reading GGUF header from {file_name}: {e}")
        return None


def parse_gguf_model(file_name: str):
    parser = Llama(model_path=file_name, vocab_only=True, verbose=False)
    return parser


def extract_gguf_metadata(file_name: str, id_values=None):
    if id_values is None:
        id_values = defaultdict(dict)

    # File size for memory management
    try:
        file_size = os.path.getsize(file_name)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return {}

    id_values["file_size"] = file_size

    header_info = read_gguf_header(file_name)
    if header_info is None or header_info[0] != b"GGUF":
        print(f"Invalid GGUF magic number in '{file_name}'")
        return {}

    magic, version = header_info
    if version < 2:
        print(f"Unsupported GGUF version {version} in '{file_name}'")
        return {}

    parser = parse_gguf_model(file_name)
    if not parser:
        return id_values

    arch = parser.metadata.get("general.architecture", "")
    name = parser.metadata.get("general.name", "")
    id_values["name"] = name or arch

    id_values["dtype"] = getattr(parser.scores, 'dtype', None) and parser.scores.dtype.name  # e.g., 'float32'

    return id_values
