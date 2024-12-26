
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from math import isclose
import re
import os
import hashlib
import sys


class ExtractAndMatchMetadata:

    def is_pattern_in_layer(self, block_pattern: list | str, layer_element: list) -> bool:
        """
        Match a string, int or regex pattern to metadata (specifically state dict layers)\n
        :param block_pattern: `str` | `int` Regex patterns, strings, or number from known identifiers
        :param layer_element: `list` Values from the metadata as str or int
        :return: boolean value of match (or not)\n
        note: prep with conditional `if entry.startswith("r'")`
        """
        if layer_element == "" or layer_element is None or block_pattern is None:
            return False
        if isinstance(layer_element, str):
            if layer_element.startswith("r'"):
                # Regex conversion
                expression_pattern = (layer_element
                            .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                            .replace(".", r"\.")    # Escape literal dots with '\.'
                            .strip("r'")            # Strip the 'r' and quotes from the string
                            )
                #print(expression)
                in_parsed_layer = re.compile(expression_pattern)
                return bool(in_parsed_layer.search(block_pattern))
            else:
                return block_pattern.lower() in layer_element.lower()
        elif isinstance(block_pattern, list) and isinstance(layer_element, list):
            return block_pattern == layer_element
        elif isinstance(block_pattern, int) and isinstance(layer_element, int):
            if block_pattern == layer_element or isclose(block_pattern, layer_element, rel_tol=1e-1):
                return True
        return False

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute and return the SHA256 hash of a given file.\n
        :param file_path: `str` Valid path to a file
        :return: `str` Hexadecimal representation of the SHA256 hash.
        :raises FileNotFoundError: File does not exist at the specified path.
        :raises PermissionError: Insufficient permissions to read the file.
        :raises IOError:  I/O related errors during file operations.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        else:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
