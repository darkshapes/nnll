### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
from pathlib import Path


def coordinate_header_tools(file_path_named: str, file_extension: str) -> tuple:
    """
    Detect file type and skim metadata from a model file using the appropriate tools\n
    :param file_path_named: `str` The full path to the file being analyzed
    :return: `tuple` Four values constituting the metadata header and external file attributes\n
    (model_header, disk_size, file_name, file_extension)
    """
    import itertools

    from nnll_04 import ModelTool

    # todo: move this and import statements to .json file
    safetensors_extensions = [".safetensors", ".sft"]

    gguf_extensions = [".gguf"]

    pickletensor_extensions = [
        ".pt",
        ".pth",
        ".ckpt",
    ]

    supported_extensions = itertools.chain(safetensors_extensions, gguf_extensions, pickletensor_extensions)

    if file_extension == "" or file_extension is None or file_extension not in list(supported_extensions):  # Skip file if we cannot possibly know what it is
        return
    else:
        model_tool = ModelTool()
        if file_extension in safetensors_extensions:
            open_header_method = model_tool.metadata_from_safetensors
        elif file_extension in gguf_extensions:
            open_header_method = model_tool.metadata_from_gguf
        elif file_extension in pickletensor_extensions:
            open_header_method = model_tool.metadata_from_pickletensor

        return open_header_method
