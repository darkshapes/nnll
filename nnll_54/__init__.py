### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from pathlib import Path
from nnll_04 import metadata_from_safetensors
from nnll_05 import metadata_from_gguf
from nnll_28 import metadata_from_pickletensor


class ModelTool:
    """Output state dict from a model file at [path]"""

    def __init__(self):
        self.read_method = None

    def read_metadata_from(self, file_path_named: str) -> dict:
        """
        Detect file type and skim metadata from a model file using the appropriate tools\n
        :param file_path_named: `str` The full path to the file being analyzed
        :return: `dict` a dictionary including the metadata header and external file attributes\n
        (model_header, disk_size, file_name, file_extension)
        """
        metadata = None
        extension = Path(file_path_named).suffix
        import_map = {
            ".safetensors": metadata_from_safetensors,
            ".sft": metadata_from_safetensors,
            ".gguf": metadata_from_gguf,
            ".pt": metadata_from_pickletensor,
            ".pth": metadata_from_pickletensor,
            ".ckpt": metadata_from_pickletensor,
        }
        if extension in import_map:
            self.read_method = import_map.get(extension)
            metadata = self.read_method(file_path_named)
        else:
            nfo(f"Unsupported file extension: {extension}")
        return metadata
