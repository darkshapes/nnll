# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Load model metadata"""

# pylint: disable=import-outside-toplevel
from pathlib import Path
from typing import Dict
import os
from nnll.configure.constants import ExtensionType
from nnll.metadata.json_io import write_json_file
from nnll.monitor.console import nfo
from nnll.monitor.file import dbug


class ReadModelTags:
    """Output state dict from a model file"""

    GGUF_MAGIC_NUMBER = b"GGUF"

    def __init__(self):
        pass

    def attempt_all_open(self, file_path_named: str, separate_desc: bool = True) -> dict | None:
        """Try all methods of extracting the metadata from the model until one succeeds\n
        *Be certain this function is only provided trusted model files to open*
        :param file_path_named: The full path to the file being opened
        :param separate_desc: Include `__metadata__` tag, or exclude and return only tensor layer names and shapes
        :return: A `dict` with the header data prepared to read, or `None`"""
        from sys import modules as sys_modules

        meta_gguf = (
            self.create_gguf_reader,
            self.create_llama_parser,
        )
        meta_safe = (
            self.metadata_from_safetensors,
            self.metadata_from_safe_open,
        )
        meta_onnx = (
            self.metadata_from_onnx_rt,
            self.metadata_from_onnx,
        )
        meta_combined = (
            self.meta_load_pickletensor,
            self.meta_load_pickletensor,
        )
        file_extension = file_path_named.split(".")[-1]

        if file_extension not in [*ExtensionType.SAFE, *ExtensionType.ONNX, *ExtensionType.PICK, *ExtensionType.MEDIA, ".py"] and self.gguf_check(file_path_named):
            for read_method in meta_gguf:
                metadata = read_method(file_path_named)
            if metadata and len(metadata) > 1:
                return metadata
        if file_extension not in [*ExtensionType.GGUF, *ExtensionType.ONNX, *ExtensionType.PICK, *ExtensionType.MEDIA, ".py"] and "pytest" not in sys_modules:
            for read_method in meta_safe:
                metadata = read_method(file_path_named, separate_desc)
                if metadata and len(metadata) > 1:
                    return metadata
        if file_extension not in [*ExtensionType.GGUF, *ExtensionType.SAFE, *ExtensionType.PICK, *ExtensionType.SCHEMA, *ExtensionType.MEDIA, ".py"]:
            for read_method in meta_onnx:
                metadata = read_method(file_path_named, separate_desc)
                if metadata and len(metadata) > 1:
                    return metadata
        if file_extension not in [*ExtensionType.GGUF, *ExtensionType.SAFE, *ExtensionType.ONNX, *ExtensionType.MEDIA, ".py"]:
            for read_method in meta_combined:
                metadata = read_method(file_path_named)
                if metadata and len(metadata) > 1:
                    return metadata
        return None

    def attempt_file_open(self, file_path_named: str, separate_desc: bool) -> dict:
        """Try two methods of extracting the metadata from the model\n
        :param file_path_named: The full path to the file being opened
        :type file_path_named: str
        :param separate_desc: Include `__metadata__` tag, or exclude and return only tensor layer names and shapes
        :type separate_desc str
        :return: A `dict` with the header data prepared to read
        """
        from sys import modules as sys_modules

        def attempt_metadata(extraction_func, fallback_func=None):
            """Attempt to extract metadata using extraction_func.
            If it fails or results in minimal metadata, use fallback_func if provided."""
            nonlocal metadata
            metadata = extraction_func()
            if not metadata or len(metadata) == 1 and fallback_func:
                metadata = fallback_func()

        metadata = None
        file_extension = Path(file_path_named).suffix

        if file_extension in ExtensionType.SAFE:
            attempt_metadata(
                lambda: self.metadata_from_safetensors(file_path_named, separate_desc),
                lambda: self.metadata_from_safe_open(file_path_named, separate_desc),
            )
        elif file_extension in ExtensionType.GGUF:
            if self.gguf_check(file_path_named):
                attempt_metadata(
                    lambda: self.create_gguf_reader(file_path_named),
                    lambda: self.create_llama_parser(file_path_named),
                )
        elif file_extension in ExtensionType.PICK:
            attempt_metadata(
                lambda: self.meta_load_pickletensor(file_path_named),
                lambda: self.meta_load_pickletensor(file_path_named),
            )
        elif file_extension in ExtensionType.ONNX and "pytest" not in sys_modules:
            attempt_metadata(
                lambda: self.metadata_from_onnx(file_path_named, separate_desc),
                lambda: self.metadata_from_onnx_rt(file_path_named, separate_desc),
            )

        return metadata

    def read_metadata_from(self, file_path_named: str, separate_desc: bool = True) -> Dict[str, str]:
        """Detect file type and skim metadata from a model file using the appropriate tools\n
        This is the input method for this class.\n
        :param file_path_named: `str` The full path to the file being analyzed
        :return: `dict` a dictionary including the metadata header and external file attributes\n
        (model_header, disk_size, file_name, file_extension)"""

        extension = Path(file_path_named).suffix.lower()

        if not any(extension in ext_type for ext_type in ExtensionType.MODEL if extension):
            dbug("Unsupported file extension: %s", f"{extension}. Silently ignoring")
        else:
            metadata = self.attempt_file_open(file_path_named, separate_desc=separate_desc) or None
            if metadata is None:
                nfo(f"Couldn't load model metadata for {file_path_named}")
                return None
            return metadata

    def gguf_check(self, file_path_named: str) -> bool:
        """A magic word check to ensure a file is GGUF format\n
        :param file_path_named: `str` the full path to the file being opened
        :return: `bool' Success or failure condition of the check"""

        import struct

        try:
            with open(file_path_named, "rb") as file_contents_to:
                magic_number = file_contents_to.read(4)
                version = struct.unpack("<I", file_contents_to.read(4))[0]
        except ValueError as error_log:
            dbug("Error reading GGUF header from %s", f"{file_path_named}: {error_log}", tb=error_log.__traceback__)
        else:
            if not magic_number and magic_number != self.GGUF_MAGIC_NUMBER:
                dbug("Invalid GGUF magic number in %s", file_path_named)
                result = False
            elif version < 2:
                dbug("Unsupported GGUF version %s", version, file_path_named)
                result = False
            elif magic_number == self.GGUF_MAGIC_NUMBER and version >= 2:
                result = True
            else:
                result = False
        return result

    def create_gguf_reader(self, file_path_named: str) -> dict:
        """Attempt to open gguf file with method from gguf library\n
        :param file_path_named: Absolute path to the file being opened
        :type file_path_named: `str`
        :return: `dict` of relevant data from the file"""
        try:
            from gguf import GGUFReader
        except (ImportError, ModuleNotFoundError):  # as error_log:
            dbug("'gguf' llibrary not available")

        try:  # method using gguf library, better for LDM conversions
            reader = GGUFReader(file_path_named, "r")  # obsolete in numpy 2, also slower
        except (UnboundLocalError, ValueError, AttributeError) as error_log:
            dbug("Value error assembling GGUFReader >:V %s", error_log, tb=error_log.__traceback__)
        else:
            reader_data = None
            arch = reader.fields.get("general.architecture")  # model type
            if arch and hasattr(arch, "parts"):
                reader_data = {
                    "architecture_name": str(bytes(arch.parts[arch.data[0]]), encoding="utf-8"),
                }
            general_name_raw = reader.fields.get("general.name")
            if general_name_raw:
                try:
                    general_name_data = general_name_raw.parts[general_name_raw.data[0]]
                    general_name = (str(bytes(general_name_data), encoding="utf-8"),)
                except KeyError as error_log:
                    dbug(
                        "Failed to get expected field within model metadata: %s",
                        file_path_named,
                        general_name_raw,
                        error_log,
                        tb=error_log.__traceback__,
                    )
                else:
                    reader_data.setdefault("general_name", general_name)
            # retrieve model name from the dict data
            tensor_data = {
                "dtype": reader.data.dtype.name,
                "types": arch.types if arch and hasattr(arch, "types") else "",
            }
            # get dtype from metadata here
            for tensor in reader.tensors:
                tensor_info = {"shape": str(tensor.shape), "dtype": str(tensor.tensor_type.name)}
                tensor_data.setdefault(str(tensor.name), tensor_info)  # safetensors normalization
            file_metadata = reader_data, tensor_data
            return file_metadata

    def create_llama_parser(self, file_path_named: str) -> dict:
        """Llama handler for gguf file header\n
        :param file_path_named: `str` the full path to the file being opened
        :return: `dict` The entire header with Llama parser formatting"""
        from llama_cpp import Llama

        try:
            parser = Llama(model_path=file_path_named, vocab_only=True, verbose=False)
        except (UnboundLocalError, ValueError, AttributeError) as error_log:
            dbug("Value error assembling GGUFReader >:V %s", error_log, tb=error_log.__traceback__)
        else:
            if parser:
                llama_data = {}

                # Extract the name from metadata using predefined keys
                name_keys = [
                    "general.basename",
                    "general.base_model.0",
                    "general.name",
                    "general.architecture",
                ]
                try:
                    for key in name_keys:
                        value = parser.metadata.get(key)
                        if value is not None:
                            llama_data.setdefault("name", value)
                            break

                    # Determine the dtype from parser.scores.dtype, if available
                    scores_dtype = getattr(parser.scores, "dtype", None)
                    if scores_dtype is not None:
                        llama_data.setdefault("dtype", scores_dtype.name)  # e.g., 'float32'
                    # file_metadata = {UpField.METADATA: llama_data, DownField.LAYER_DATA: EmptyField.PLACEHOLDER}
                except ValueError as error_log:
                    dbug("Parsing file failed for %s", file_path_named, error_log, tb=error_log.__traceback__)

            return llama_data

    def metadata_from_pickletensor(self, file_path_named: str) -> dict:
        """Collect metadata from a pickletensor file header\n
        :param file_path: `str` the full path to the file being opened
        :return: `dict` the key value pair structure found in the file"""
        import pickle
        from mmap import mmap

        # separate_desc not implemented here yet
        with open(file_path_named, "r+b") as file_contents_to:
            mem_map_data = mmap(file_contents_to.fileno(), 0)
            view = memoryview(mem_map_data)
            return pickle.loads(view)

    def meta_load_pickletensor(self, file_path_named: str, separate_desc: bool = True) -> dict:
        """Load metadata from a pickled tensor file.\n
        **USE ONLY FOR TRUSTED FILES**
        :param file_path_named: Path to the pickled tensor file containing metadata.
        :type file_path_named: str
        :return: A dictionary containing the loaded metadata from the file.
        :rtype: dict"""

        import torch

        torch.set_num_threads(1)
        from _pickle import UnpicklingError

        try:
            metadata = torch.load(file_path_named, map_location="meta")
        except UnpicklingError:
            return
        else:
            if separate_desc:  # in this case we leave the rest of the dict behind
                meta_copy = metadata.copy()
                meta_key = next(iter(metadata))
                if isinstance(meta_copy.get(meta_key), dict):
                    try:
                        metadata = {meta_key: {"dtype": tensor.dtype, "shape": tensor.size, "data_offsets": tensor.storage_offset} for meta_key, tensor in meta_copy.get(meta_key).items()}
                    except AttributeError:
                        pass
                    else:
                        return metadata

    def metadata_from_safetensors(self, file_path_named: str, separate_desc: bool = True) -> dict:
        """
        Collect metadata from a safetensors file header\n
        :param file_path_named: `str` the full path to the file being opened
        :return: `dict` the key value pair structure found in the file
        """
        import json
        import struct

        assembled_data = {}
        with open(file_path_named, "rb") as file_contents_to:
            try:
                length_of_header = struct.unpack("<Q", file_contents_to.read(8))[0]
                header_data = file_contents_to.read(length_of_header)
                header_data = json.loads(header_data.decode("utf-8", errors="strict"))
            except (json.JSONDecodeError, MemoryError) as error_log:
                dbug("Failed to read json from file : %s", file_path_named, error_log, tb=error_log.__traceback__)
            else:
                if separate_desc:
                    try:
                        assembled_data = header_data.copy()
                        assembled_data.pop("__metadata__", assembled_data)
                        header_data = assembled_data
                    except KeyError as error_log:
                        dbug("Couldnt remove '__metadata__' from header data. %s", header_data, error_log, tb=error_log.__traceback__)
                return header_data

    def metadata_from_safe_open(self, file_path_named: str, separate_desc: bool = True) -> dict:
        """
        Collect metadata from a safetensors file header.\n
        This method is less performant than `struct`\n
        :param file_path_named: `str` the full path to the file being opened
        :return: `dict` the key value pair structure found in the file
        """
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(file_path_named, framework="pt", device="cpu") as layer_content:
                file_data = {key: layer_content.get_tensor(key).shape for key in layer_content}
                metadata = file_data | layer_content.metadata() if not separate_desc else file_data
                # if not separate_desc:
                #     metadata.update(layer_content.metadata())
            return metadata

        except (SafetensorError, TypeError):
            pass

    def metadata_from_onnx_rt(self, file_path_named: str, separate_desc: bool = True) -> Dict[str, str]:
        """Extract metadata from an ONNX model using ONNX Runtime.\n
        :param file_path_named: The path to the ONNX model file
        :param separate_desc: Exclude or include metadata description, default True
        :return: A mapping of metadata about the model
        """
        from onnxruntime import InferenceSession  # , get_available_providers
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NotImplemented, Fail, RuntimeException

        try:
            onnx_sess = InferenceSession(file_path_named)
            meta = onnx_sess.get_modelmeta()
        except (InvalidProtobuf, NotImplemented, Fail, RuntimeException) as error_log:
            dbug(error_log)
            pass
        else:
            if separate_desc:
                return {"custom_metadata_map": meta.custom_metadata_map}
            else:
                {tag: getattr(meta, tag, {}) for tag in dir(meta) if not tag.startswith("_")}

    def metadata_from_onnx(self, file_path_named: str, separate_desc: bool = True) -> Dict[str, str]:
        """Extract metadata from an ONNX model using the ONNX library.\n
        :param file_path_named: The path to the ONNX model file
        :param separate_desc: Exclude or include metadata description, default True
        :return: A mapping of metadata about the model
        """
        from onnxruntime.datasets import get_example
        from onnx import load as onnx_load
        from google.protobuf.message import DecodeError
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf

        try:
            example = get_example(file_path_named)
            model = onnx_load(example)
        except (InvalidProtobuf, DecodeError) as error_log:
            dbug(error_log)
            pass
        else:
            if separate_desc:
                return {"metadata_props": model.metadata_props}
            else:
                return {tag: getattr(model, tag, {}) for tag in dir(model) if not tag.startswith("_")}


def main(
    folder_path_named: str | None = None,
    save_location: str | None = os.getcwd(),
    separate_desc: bool | None = None,
    unsafe: bool | None = None,
) -> None:
    import argparse
    from sys import modules as sys_modules

    from nnll.integrity import ensure_path

    if "pytest" not in sys_modules:  # bypass in case of testing
        # Set up argument parser
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scan the state dict metadata from a folder of files at [path] to the console,\
                 then write to a json file at [save]\nOffline function.",
            usage="nnll-meta ~/Downloads/models/images -s ~Downloads/models/metadata",
            epilog=f"Valid input formats: {[*ExtensionType.MODEL]}",
        )
        parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=os.getcwd())
        parser.add_argument("-s", "--save_to_folder_path", required=False, help="Path where output should be stored. (default: '.')", type=str, default=os.getcwd())
        parser.add_argument("-d", "--separate_desc", required=False, action="store_true", help="Ignore the metadata from the header. (default: False)", default=False)
        parser.add_argument("-u", "--unsafe", action="store_true", help="Try to read non-standard type model files. MAY INCLUDE NON-MODEL FILES. (default: False)")
        args = parser.parse_args()
    else:
        args = None

    folder_path_named = os.getcwd() if not args else args.path
    separate_desc = True if not args else args.separate_desc
    save_location = save_location if not args else args.save_to_folder_path
    unsafe = False if not args else args.unsafe
    reader = ReadModelTags()
    if folder_path_named is not None:
        for root, folders, files in os.walk(folder_path_named):
            for file_name in files:
                file_path_named = os.path.join(root, file_name)
                if not unsafe:
                    metadata = reader.read_metadata_from(file_path_named, separate_desc=separate_desc)
                else:
                    metadata = reader.attempt_all_open(file_path_named, separate_desc=separate_desc)
                if metadata is not None:
                    save_location = ensure_path(save_location)
                    write_json_file(save_location, f"{os.path.dirname(root)}_{os.path.basename(root)}_{file_name}.json", metadata)
