# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s


import os
import struct
from gguf import GGUFReader
from llama_cpp import Llama

GGUF_MAGIC_NUMBER = b"GGUF"


def gguf_check(file_path_named: str) -> tuple:
    """
    A magic word check to ensure a file is GGUF format\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `tuple' the number
    """

    try:
        with open(file_path_named, "rb") as file_contents_to:
            magic_number = file_contents_to.read(4)
            version = struct.unpack("<I", file_contents_to.read(4))[0]
            print(version)
    except ValueError as error_log:
        print(f"Error reading GGUF header from {file_path_named}: {error_log}")
    else:
        if not magic_number and magic_number != GGUF_MAGIC_NUMBER:
            print(f"Invalid GGUF magic number in '{file_path_named}'")
            result = False
        elif version < 2:
            print(f"Unsupported GGUF version {version} in '{file_path_named}'")
            result = False
        elif magic_number == GGUF_MAGIC_NUMBER and version >= 2:
            result = True
        else:
            result = False
    return result


def create_gguf_reader(file_path_named: str) -> dict:
    """
    Attempt to open gguf file with method from gguf library\n
    :param file_path_named: Absolute path to the file being opened
    :type file_path_named: `str`
    :return: `dict` of relevant data from the file
    """

    try:  # method using gguf library, better for LDM conversions
        reader = GGUFReader(file_path_named, "r")  # obsolete in numpy 2, also slower
    except ValueError as error_log:
        print(error_log)  # >:V
    else:
        arch = reader.fields.get("general.architecture")  # model type
        reader_data = {
            "architecture_name": str(bytes(arch.parts[arch.data[0]]), encoding="utf-8"),
        }
        general_name_raw = reader.fields.get("general.name")
        if general_name_raw:
            try:
                general_name_data = general_name_raw.parts[general_name_raw.data[0]]
                general_name = (str(bytes(general_name_data), encoding="utf-8"),)
            except KeyError as error_log:
                print(
                    "Failed to get expected field within model metadata: %s",
                    file_path_named,
                    general_name_raw,
                    error_log,
                )
            else:
                reader_data.setdefault("general_name", general_name)
        # retrieve model name from the dict data
        tensor_data = {
            "dtype": reader.data.dtype.name,
            "types": arch.types if len(arch.types) > 1 else "",
        }
        # get dtype from metadata here
        for tensor in reader.tensors:
            tensor_info = {"shape": str(tensor.shape), "dtype": str(tensor.tensor_type.name)}
            tensor_data.setdefault(str(tensor.name), tensor_info)  # safetensors normalization
        file_metadata = reader_data, tensor_data
        return file_metadata


def create_llama_parser(file_path_named: str) -> dict:
    """
    Llama handler for gguf file header\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `dict` The entire header with Llama parser formatting
    """
    file_metadata = {}
    parser = Llama(model_path=file_path_named, vocab_only=True, verbose=False)
    if parser:
        llama_data = {}

        # Extract the name from metadata using predefined keys
        name_keys = [
            "general.basename",
            "general.base_model.0",
            "general.name",
            "general.architecture",
        ]
        for key in name_keys:
            value = parser.metadata.get(key)
            if value is not None:
                llama_data.setdefault("name", value)
                break

        # Determine the dtype from parser.scores.dtype, if available
        scores_dtype = getattr(parser.scores, "dtype", None)
        if scores_dtype is not None:
            llama_data.setdefault("dtype", scores_dtype.name)  # e.g., 'float32'
        file_metadata = llama_data

    return file_metadata


def attempt_file_open(file_path_named: str) -> dict:
    """
    Try two methods of extracting the metadata from the file\n
    :param file_path_named: The full path to the file being opened
    :type file_path_named: str
    :return: A `dict` with the header data prepared to read
    """
    metadata = create_gguf_reader(file_path_named)
    if metadata:
        pass
    else:
        try:
            metadata = create_llama_parser(file_path_named)
        except ValueError as error_log:
            print("Parsing .gguf file failed for %s", file_path_named, error_log)
    return metadata


def metadata_from_gguf(file_path_named: str) -> dict:
    """
    Collect metadata from a gguf file header\n
    :param file_path_named: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """

    if gguf_check(file_path_named):
        file_metadata = attempt_file_open(file_path_named)
        if file_metadata is not None:
            return file_metadata
