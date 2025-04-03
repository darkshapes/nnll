### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from nnll_01 import debug_monitor, info_message as nfo
from nnll_60 import JSONCache, HASH_PATH_NAMED

cache_manager = JSONCache(HASH_PATH_NAMED)


def hash_layers_or_files(folder_path: str, mode: str = "layer"):
    import os
    from nnll_04 import ModelTool
    from nnll_44 import compute_hash_for
    from pathlib import Path

    model_tool = ModelTool()
    nfo(folder_path)
    hash_values = {}

    folder_contents = os.listdir(os.path.normpath(Path(folder_path)))
    for file_name in folder_contents:
        if Path(file_name).suffix.lower() in [".safetensors", ".sft", ".gguf"]:
            file_path_named = os.path.join(folder_path, file_name)
            file_size = os.path.getsize(file_path_named)  # 1GB
            if mode != "layer" or file_size < 1e9:
                hex_value = compute_hash_for(file_path_named=file_path_named)
                hash_values.setdefault(hex_value, file_path_named)
                nfo(f"'{file_name}' : '{hex_value}'")
            else:
                state_dict = model_tool.read_metadata_from(file_path_named)
                hex_value = compute_hash_for(text_stream=str(state_dict))
                hash_values.setdefault(hex_value, file_path_named)
                nfo(f"'{hex_value}' : '{file_name}'")
    return hash_values


def check_model_identity(known_hash: dict, hex_value: str, attributes: dict | None = None) -> bool:
    """
    Iteratively structure unpacked hash values into reference pattern, then feed a equivalence check.\n
    :param known_hash: `dict` A dictionary of hash values known to identify models
    :param unpacked_metadata: `dict` Values from the unknown files
    :param attributes: `dict` Optional additional metadata, such as tensor count and file_size (None will bypass necessity of these matches)
    :return: `bool` Whether or not the values from the model header and tensors were found inside pattern_details\n
    """
    for mir_name, data in known_hash.items():
        nfo(f"islist, {type(data)}")
        if isinstance(data, list):
            for known in data:
                nfo(f"{hex_value} == {known} ??")
                if known == hex_value:
                    return mir_name


@cache_manager.decorator
def compare_hash_values(hash_values: dict, data: dict):
    import os
    from tqdm.auto import tqdm

    model_id = {}
    for hex_value, file_path_named in tqdm(hash_values.items(), total=len(hash_values), position=0, leave=True):
        known_hash = ""
        if os.path.getsize(file_path_named) > 1e9:  # 1GB
            known_hash = data.get("layer_256")
        if not known_hash:
            known_hash = data.get("file_256")
        trail = check_model_identity(known_hash, hex_value)
        nfo(trail)
        if trail:
            model_id.setdefault(hex_value, trail)
    return model_id


# put matching keys into index folder
# tqdm it


@debug_monitor
def main():
    """Parse arguments to feed to dict header reader"""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Output the hash of a state dict or file from all model files at [path] to the console", epilog="Example: nnll-hash '~/Downloads/models/images'")
    parser.add_argument("-m", "--mode", help="Change mode to calculate hash for the whole file", action="store_true")
    parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=".")

    args = parser.parse_args()
    if args.mode:
        expression = {"path": args.path, "mode": "file"}
    else:
        expression = {"path": args.path, "mode": "layer"}

    hash_values = hash_layers_or_files(**expression)
    nfo(hash_values)


if __name__ == "__main__":
    main()
