### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import nnll_01
from nnll_01 import debug_monitor, info_message as nfo


def hash_layers(path: str, mode: str = "layer"):
    import os
    from nnll_04 import ModelTool
    from nnll_44 import compute_hash_for
    from pathlib import Path
    from tqdm.auto import tqdm

    model_tool = ModelTool()
    nfo(path)
    hash_values = {}

    folder_contents = os.listdir(os.path.normpath(Path(path)))
    for file_name in tqdm(folder_contents, total=len(folder_contents), position=0, leave=True):
        if Path(file_name).suffix.lower() in [".safetensors", ".sft", ".gguf"]:
            if mode != "layer":
                hash_values.setdefault(file_name, compute_hash_for(file_path_named=os.path.join(path, file_name)))
            else:
                state_dict = model_tool.read_metadata_from(os.path.join(path, file_name))
                hash_values.setdefault(compute_hash_for(text_stream=str(state_dict)))
                nfo(f"'{hash_values}' : '{file_name}'")
        return hash_values


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

    hash_values = hash_layers(**expression)
    nfo(hash_values)


if __name__ == "__main__":
    main()
