# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from types import FunctionType
from typing import Dict, List, Union
from array import array


async def sum_layers_of(model_metadata: str, hash_function: FunctionType) -> str:
    """Compute hashes for a list of model layers.\n
    :param model_metadata: Metadata from the model layers.
    :param hash_function: An existing instance of ReadModelTags.
    :return: A string representing the hashes of the concatenated layers names."""

    if model_metadata and isinstance(model_metadata, tuple):
        model_metadata = next(iter(metadata for metadata in model_metadata if metadata.get("types", False)), model_metadata)
    model_metadata = list(model_metadata)
    layer_data = "".join(model_metadata)
    return await hash_function(text_stream=layer_data)


async def sum_models_in(folder_path_named: str, layers: bool = True, b3_sum: bool = True) -> Dict[str, str]:
    """Compute BLAKE3 hashes for models in a given folder.

    This function reads all model files within the specified folder, computes the BLAKE3 hash for each model's layers,
    and returns a dictionary mapping each model file path to its corresponding hash.

    :param folder_path_named: The path to the folder containing the model files.
    :return: A dictionary where keys are the file paths of the models and values are their respective BLAKE3 hashes.
    """

    import os
    from nnll.metadata.model_tags import ReadModelTags
    from nnll.configure.constants import ExtensionType
    from pathlib import Path

    if b3_sum:
        from nnll.integrity.hashing import compute_b3_for as hash_function
    else:
        from nnll.integrity.hashing import compute_hash_for as hash_function
    tag_reader = ReadModelTags()
    model_hashes = {}

    for file_name in os.listdir(folder_path_named):
        extension = Path(file_name).suffix.lower()
        if any(extension in ext_type for ext_type in ExtensionType.MODEL if extension):
            file_path_named = os.path.join(folder_path_named, file_name)
            if os.path.isfile(file_path_named):
                if layers:
                    model_metadata: Union[List[str], Dict[str, Dict[str, array]]] = tag_reader.read_metadata_from(file_path_named)
                    layer_hash = await sum_layers_of(model_metadata, tag_reader=tag_reader, hash_function=hash_function)
                    model_hashes[file_name] = layer_hash
                else:
                    file_hash = await hash_function(file_path_named=file_path_named)
                    model_hashes[file_name] = file_hash

    return model_hashes
