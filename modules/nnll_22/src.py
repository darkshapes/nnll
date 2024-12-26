
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
from abc import ABC, abstractmethod
from typing import Union, List


class AbstractLink(ABC):
    """
    ####  Valid subclasses
    #### `Autoencoder`: [filename] diffusion_pytorch_model [folder_name] vae
    #### `TextEncoderLink`:  [filename] model [folder_name] text_encoder
    #### `UNetLink`:  [filename] diffusion_pytorch_model [folder_name] unet
    """

    def __init__(self, metadata_folder: str = "/Users/unauthorized/Downloads/models/metadata"):
        self.metadata_folder = os.path.normpath(metadata_folder)

    @abstractmethod
    def get_filename(self) -> str:
        pass

    @abstractmethod
    def get_folder_name(self, index: int) -> str:
        pass

    def create_symlink( self, model_type: str, target_path: str | List[str] = None,
                        original_layout: bool = False, variant_suffix: str = "safetensors"
                        ) -> str | List[str]:
        """
        #### Create a symlink from a file in a model database
        #### `model_type`: Id code of model
        #### `target_path`: Original file or a list of sharded files to symlink
        #### `original_layout`: Format the symlink like the original repository
        #### `variant_suffix`: Adjust the suffix/extension for fp16 or other formats
        #### OUTPUT: symlinks of target files at the specified locations
        """
        filename = f"{self.get_filename()}.{variant_suffix}"  # Modify the filename by appending or replacing the suffix
        folder_name = os.path.join(model_type, self.get_folder_name(1)) if original_layout else model_type

        def _add_link(target: str, filename: str) -> str:
            """
            #### Iterable symlinker core method
            #### `target`: The file or folder that gets symlinked
            #### `filename`: The name of the symlink pointing to the target
            #### OUTPUT: A path to a symlink
            """
            symlink_full_path = os.path.join(self.metadata_folder, folder_name, filename)
            os.makedirs(os.path.dirname(symlink_full_path), exist_ok=True)   # Ensure the target directory exists
            for file_name in os.listdir(os.path.dirname(symlink_full_path)):  # Remove any existing symlink or file at the destination path
                if os.path.islink(os.path.join(os.path.dirname(symlink_full_path), file_name)):
                    os.remove(os.path.join(os.path.dirname(symlink_full_path), file_name))

            os.symlink(target, symlink_full_path)  # Create the new symlink
            return symlink_full_path

        if target_path is not None:
            if isinstance(target_path, list):  # Handle sharded files
                symlink_locations = []
                split_filename = filename.split(".")
                for i, shard in enumerate(target_path):
                    # zfill 5 means 9 shards supported at max, current hf limit being 50gb this means ~441gb max model size
                    filename = f"{split_filename[0]}-{str(i + 1).zfill(5)}-of-{str(len(target_path)).zfill(5)}.{split_filename[1]}"
                    shard_path = _add_link(shard, filename)
                    symlink_locations.append(shard_path)
                return symlink_locations

            elif isinstance(target_path, str):  # Single file handling
                symlink_location = _add_link(target_path, filename)
                return symlink_location
            else:
                raise ValueError("Either 'target_path' or 'sharded_files' must be provided.")

        raise ValueError("Either target_path or sharded_files must be provided")

# Specific link classes


class AutoencoderLink(AbstractLink):
    def get_filename(self) -> str:
        return "diffusion_pytorch_model"

    def get_folder_name(self, index: int) -> str:
        return f"vae_{index}" if index > 1 else "vae"


class TextEncoderLink(AbstractLink):
    def get_filename(self) -> str:
        return "model"

    def get_folder_name(self, index: int) -> str:
        return f"text_encoder_{index}" if index > 1 else "text_encoder"


class UNetLink(AbstractLink):
    def get_filename(self) -> str:
        return "diffusion_pytorch_model"

    def get_folder_name(self, index: int) -> str:
        return f"unet_{index}" if index > 1 else "unet"
