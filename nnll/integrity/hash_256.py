# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import os
import sys
from types import FunctionType
from pathlib import Path

sys.path.append(os.getcwd())

from nnll.metadata.json_io import read_json_file

tag_file = False


class HexSum:
    """:param b3: If True, uses BLAKE3 algorithm; if False, uses SHA256. default True
    :param desc_process: Procedures metadata included in output file. default True
    :param layer: If True, processes plaintext state dict layer names; if False, processes binary files, default True
    :param quiet: Use console output if False. default False
    :param unsafe: Attempt to read metadata, even from non-normal files. default False\n
    **Only use `unsafe` on trusted cache directories!**"""

    sha = False
    desc_process = True
    layer = True
    quiet = not desc_process
    unsafe = False

    def __init__(self):
        from nnll.metadata.model_tags import ReadModelTags

        self.model_tool = ReadModelTags()

    async def hash_for_mir(self, file_path_named: str, hash_calculator: FunctionType, model_reader: FunctionType) -> tuple[str]:
        """Generate a hash for the specified file path using the provided hash calculator and model reader.\n
        :param file_path_named: Absolute path to the file.
        :param hash_calculator: A function that calculates the hash of a given file or text stream.
        :return: A tuple containing the relative file name and its corresponding hash value."""

        if self.quiet:
            from nnll.monitor.file import dbug as nfo
        else:
            from nnll.monitor.console import nfo

        file_name = os.path.basename(file_path_named)
        folder_path_named = os.path.basename(os.path.dirname(file_path_named))
        if os.path.isdir(file_path_named):
            return
        if self.layer is False:
            hex_value = await hash_calculator(file_path_named=file_path_named)
            nfo(f"'{file_name}' : '{os.path.basename(hex_value)}'")
            return os.path.join(folder_path_named, file_name), hex_value

        else:
            try:
                state_dict = model_reader(file_path_named, separate_desc=self.layer)
            except Exception as error_log:
                nfo(error_log)
                return
            hex_value = await hash_calculator(text_stream=str(state_dict))
            nfo(f"'{hex_value}' : '{file_name}'")  # reversed to highlight difference
            return os.path.join(folder_path_named, file_name), hex_value

    async def sum_models_in(self, folder_contents: list[str], folder_path: str, hash_calculator: FunctionType, hash_values: dict) -> dict:
        """Loop calculation of hash totals for models in a folder
        :param folder_contents: The file contents of a folder
        :param folder_path: Location of the folder
        :param hash_calculator: Hash function to use
        :param hash_values: A dict to store the resulting sums
        :return: A mapping of files to their sums"""
        from nnll.configure.constants import ExtensionType
        from tqdm.asyncio import tqdm

        model_reader = self.model_tool.attempt_all_open if self.unsafe else self.model_tool.read_metadata_from
        async for file_name in tqdm(folder_contents, total=len(folder_contents), position=-2, leave=True, disable=self.quiet):
            if any(Path(file_name).suffix.lower() in extensions for extensions in ExtensionType.MODEL) or self.unsafe:
                file_path_named = os.path.join(folder_path, file_name)
                if not any(media for media in ExtensionType.IGNORE if media in file_path_named):
                    output = await self.hash_for_mir(
                        file_path_named=file_path_named,
                        hash_calculator=hash_calculator,
                        model_reader=model_reader,
                    )
                    if output:
                        hash_values.setdefault(output[0], output[1])
        return hash_values

    async def hash_layers_or_files(self, path_named: str, layer=True, sha=False, desc_process=False, unsafe=False) -> dict[str, str]:
        """Hashes layers or files in a folder using BLAKE3 or SHA256, with optional metadata inclusion.\n
        :param path_named: Path to a filea or folder containing model files.
        :return: Dictionary mapping hash values to file names."""
        self.layer = layer
        self.sha = sha
        self.unsafe = unsafe
        self.desc_process = desc_process
        self.quiet = not desc_process
        import os
        from nnll.integrity.hashing import compute_b3_for, compute_hash_for

        if self.quiet:
            from nnll.monitor.file import dbug as nfo
        else:
            from nnll.monitor.console import nfo
        if os.path.isdir(path_named):
            folder_contents = os.listdir(os.path.normpath(Path(path_named)))
            folder_path = path_named
        else:
            folder_path = os.path.dirname(path_named)
            folder_contents = [path_named]
        hash_calculator = compute_hash_for if self.sha else compute_b3_for
        hash_values = {}
        nfo(f"{folder_path} : TYPE:{'LAYER   W METADATA: False' if self.layer else 'FILE'}   ALGORITHM:{'SHA256' if self.sha else 'BLAKE3'}")
        if not self.quiet:
            data = {
                "type": "LAYER" if self.layer else "FILE",
                "algorithm": "SHA256" if self.sha else "BLAKE3",
                "FunctionType": str(compute_hash_for.__name__).upper() if self.sha else str(compute_b3_for.__name__).upper(),
            }
            data.setdefault("with_metadata", False) if self.layer else ()
            hash_values.setdefault(folder_path, data)

        hash_values = await self.sum_models_in(folder_contents, folder_path, hash_calculator, hash_values)
        return hash_values

    def identify_model(self, database: dict[dict | list | str | int], unknown: str) -> str | None:
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

    async def compare_hash_values(self, hash_values: dict) -> list[str]:
        """Orchestrate process to determine model identifiers.\n
        :param hash_values: known hash values
        :return: a MIR tag of the model with the given hash, or None if not found"""
        from tqdm.auto import tqdm

        data = read_json_file(os.path.join(os.path.dirname(__file__), "hashes.json"))
        model_id = {}
        for hex_value, file_path_named in tqdm(hash_values.items(), total=len(hash_values), position=0, leave=True):
            known_hashes = ""
            if os.path.getsize(file_path_named) > 1e9:  # 1GB
                known_hashes = data.get("layer_256")
            if not known_hashes:
                known_hashes = data.get("file_256")
            trail = self.identify_model(known_hashes, hex_value)
            if trail:
                model_id.setdefault(hex_value, trail)
        return model_id

    async def write_to_file(self, program: FunctionType | str, hash_values: dict[str, str]) -> None:
        from datetime import datetime

        from nnll.metadata.json_io import write_json_file

        date_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        prog_name = program.replace("(", "").replace(")", "").replace("'", "")
        hash_values.setdefault(str(date_stamp), prog_name)
        write_json_file(".", f"{date_stamp}_{prog_name}.json", hash_values)


def main():
    """Parse arguments to feed to dict header reader"""
    import argparse
    import asyncio

    # Set up argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Output hashes of each model file state dict in [path] to console and .JSON \n Offline function.""",
        usage="nnll-hash '~/Downloads/models/'",
    )
    parser.add_argument("path", default=".", help="Path to the directory where files should be analyzed. (default '.'')")

    parser.add_argument("-f", "--file", action="store_true", help="Change mode to calculate hash for the whole file instead of state dict layers (default: False)")
    parser.add_argument("-s", "--sha", action="store_true", help="Change algorithm from BLAKE3 to SHA256 (default: False)")
    parser.add_argument("-d", "--describe", "--describe-process", action="store_true", help="Include processing metadata in the output (default: True)", default=True)
    parser.add_argument("-u", "--unsafe", action="store_true", help="Try to hash non-standard type model files. MAY INCLUDE NON-MODEL FILES. (default: False)")

    args = parser.parse_args()
    hex_sum = HexSum()

    hash_values = asyncio.run(
        hex_sum.hash_layers_or_files(
            path_named=args.path,
            layer=not args.file,
            sha=args.sha,
            desc_process=args.describe,
            unsafe=args.unsafe,
        ),
    )
    asyncio.run(hex_sum.write_to_file(program=str(parser.prog), hash_values=hash_values))


if __name__ == "__main__":
    main()
