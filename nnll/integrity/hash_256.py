### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->
import sys
import os

sys.path.append(os.getcwd())
from nnll.monitor.console import nfo

from nnll.metadata.json_io import read_json_file


def hash_layers_or_files(folder_path: str, layer: bool = True) -> dict[str:str]:
    """Compute model hashes from a folder, send to console and store\n
    :param folder_path: Location of model files to scan
    :param layer: Hash layer names or file contents, defaults to layer
    :return: `dict` Map of hashes(k) to filenames(v)
    """
    import os
    from nnll.metadata.model_tags import ReadModelTags
    from nnll.integrity.hashing import compute_hash_for
    from pathlib import Path

    model_tool = ReadModelTags()
    nfo(folder_path)
    hash_values = {}

    folder_contents = os.listdir(os.path.normpath(Path(folder_path)))
    for file_name in folder_contents:
        if Path(file_name).suffix.lower() in [".safetensors", ".sft", ".gguf"]:
            file_path_named = os.path.join(folder_path, file_name)
            file_size = os.path.getsize(file_path_named)  # 1GB
            if layer is False or file_size < 1e9:
                hex_value = compute_hash_for(file_path_named=file_path_named)
                hash_values.setdefault(hex_value, file_path_named)
                nfo(f"'{file_name}' : '{hex_value}'")
            else:
                state_dict = model_tool.read_metadata_from(file_path_named)
                hex_value = compute_hash_for(text_stream=str(state_dict))
                hash_values.setdefault(hex_value, file_path_named)
                nfo(f"'{hex_value}' : '{file_name}'")
    return hash_values


def identify_model(database: dict[dict | list | str | int], unknown: str) -> str | None:
    """
    Compare known values to foreign values\n
    :param database: A dictionary of content known to identify models
    :param unknown: A portion of metadata to compare to
    :param ignore_key: Optional additional metadata, such as tensor count and file_size (None will bypass necessity of these matches)
    :return: `str` Name of a matching identity, or None\n
    """
    for category in database:
        if isinstance(database.get(category), list):
            for sig in database.get(category):
                cat = [category for sig in database.get(category) if unknown in sig]
                if cat:
                    return cat

        # should be able to expand this for use with
        if isinstance(database.get(category), dict) and any((known == unknown for known in sig) for _, sig in database.get(category).items()):
            return category
    return None


def compare_hash_values(hash_values: dict):
    """
    Orchestrate process to determine model identifiers.
    :param hash_values: known hash values
    :param data: _description_
    :return: _description_
    """
    from tqdm.auto import tqdm

    data = read_json_file(os.path.join(os.path.dirname(__file__), "hashes.json"))
    model_id = {}
    for hex_value, file_path_named in tqdm(hash_values.items(), total=len(hash_values), position=0, leave=True):
        known_hashes = ""
        if os.path.getsize(file_path_named) > 1e9:  # 1GB
            known_hashes = data.get("layer_256")
        if not known_hashes:
            known_hashes = data.get("file_256")
        trail = identify_model(known_hashes, hex_value)
        nfo(trail)
        if trail:
            model_id.setdefault(hex_value, trail)
    return model_id


def main():
    """Parse arguments to feed to dict header reader"""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Output the hashes of model state dicts or files from [path] to console",
        epilog="Example: nnll-hash '~/Downloads/models/'",
    )
    parser.add_argument("-m", "--mode", help="Change mode to calculate hash for the whole file", action="store_true")
    parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=".")

    args = parser.parse_args()
    if args.mode:
        expression = {"path": args.path, "layer": False}
    else:
        expression = {"path": args.path, "layer": True}

    hash_values = hash_layers_or_files(**expression)
    nfo(hash_values)


if __name__ == "__main__":
    main()
