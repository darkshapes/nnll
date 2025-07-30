# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import os
import sys

sys.path.append(os.getcwd())

from nnll.metadata.json_io import read_json_file
from nnll.monitor.file import dbug as nfo

tag_file = False


async def hash_layers_or_files(folder_path: str, layer: bool = True, b3: bool = True, header: bool = False, unsafe: bool = False) -> dict[str, str]:
    """Hashes layers or files in a folder using BLAKE3 or SHA256, with optional metadata inclusion.\n
    :param folder_path: Path to the folder containing model files.
    :param layer: If True, processes plaintext state dict layer names; if False, processes binary files.
    :param b3: If True, uses BLAKE3 algorithm; if False, uses SHA256.
    :param header: If True, includes metadata in layer processing.
    :param unsafe: Attempt to read metadata even from non-normal files, defaults False
    :return: Dictionary mapping hash values to file names."""
    import os
    from pathlib import Path
    from tqdm.asyncio import tqdm
    from nnll.integrity.hashing import compute_hash_for
    from nnll.integrity.hashing import compute_b3_for
    from nnll.configure.constants import ExtensionType
    from nnll.monitor.file import dbug

    from nnll.metadata.model_tags import ReadModelTags

    calculate_hash = compute_b3_for if b3 else compute_hash_for
    model_tool = ReadModelTags()
    hash_values = {}
    nfo(f"{folder_path} : TYPE:{'LAYER   W METADATA: ' + str(header) if layer else 'FILE'}   ALGORITHM:{'BLAKE3' if calculate_hash == compute_b3_for else 'SHA256'}")
    if tag_file:
        data = {
            "type": "LAYER" if layer else "FILE",
            "algorithm": "BLAKE3" if calculate_hash == compute_b3_for else "SHA256",
            "FunctionType": str(compute_b3_for.__name__).upper() if b3 else str(compute_hash_for.__name__).upper(),
        }
        data.setdefault("with_metadata", str(header)) if layer else ()
        hash_values.setdefault(folder_path, data)
    folder_contents = os.listdir(os.path.normpath(Path(folder_path)))
    async for file_name in tqdm(folder_contents, total=len(folder_contents), position=-2, leave=True, disable=nfo == dbug):
        if any(Path(file_name).suffix.lower() in extensions for extensions in ExtensionType.MODEL) or unsafe:
            file_path_named = os.path.join(folder_path, file_name)
            if not any(media for media in ExtensionType.IGNORE if media in file_path_named):
                if os.path.isdir(file_path_named):
                    continue
                if layer is False:
                    hex_value = await calculate_hash(file_path_named=file_path_named)
                    hash_values.setdefault(os.path.basename(file_path_named), hex_value)
                    nfo(f"'{file_name}' : '{os.path.basename(hex_value)}'")
                else:
                    if unsafe:
                        try:
                            state_dict = model_tool.attempt_all_open(file_path_named, separate_desc=header)
                        except Exception as error_log:
                            print(error_log)
                            continue
                    else:
                        state_dict = model_tool.read_metadata_from(file_path_named, separate_desc=header)
                    hex_value = await calculate_hash(text_stream=str(state_dict))
                    hash_values.setdefault(os.path.basename(file_path_named), hex_value)
                    nfo(f"'{hex_value}' : '{os.path.basename(file_name)}'")  # reversed to highlight difference
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


async def compare_hash_values(hash_values: dict):
    """Orchestrate process to determine model identifiers.\n
    :param hash_values: known hash values
    :param data: _description_
    :return: _description_"""
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
    from sys import argv

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Output the hashes of model state dicts or files from [path] to console and file",
        epilog="Example: nnll-hash '~/Downloads/models/'",
    )
    parser.add_argument("-f", "--file", action="store_true", help="Change mode to calculate hash for the whole file instead of state dict layers")
    parser.add_argument("-s", "--sha", action="store_true", help="Change algorithm from b3 to sha")
    parser.add_argument("-i", "--include", "--include-header", action="store_true", help="Include the metadata header, only applies to hashing state dict layers")
    parser.add_argument("-u", "--unsafe", action="store_true", help="Try to hash non-standard type model files layers")
    parser.add_argument("path", default=".", help="Path to directory where files should be analyzed. (default .)")

    args = parser.parse_args()
    import asyncio
    from nnll.metadata.json_io import write_json_file
    from datetime import datetime

    tag_file = True

    program = str(parser.prog)
    location = argv[1].split(os.sep)[-2]
    hash_values = asyncio.run(hash_layers_or_files(folder_path=args.path, layer=not args.file, b3=not args.sha, header=args.include, unsafe=args.unsafe))
    date_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    hash_values[args.path][program] = str(date_stamp)
    write_json_file(".", f"{program}_{location}_{date_stamp}.json", hash_values)


if __name__ == "__main__":
    from nnll.monitor.console import nfo

    main()
