#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import re
import os
import hashlib
import sys


class ExtractAndMatchMetadata:

    def match_pattern_and_regex(self, block_pattern: str, layer_element: str) -> bool:
        """
        Match a string, int or regex pattern to metadata (specifically state dict layers)\n
        :param block_pattern: `str` | `int` Regex patterns, strings, or number from known identifiers
        :param layer_element: `str` Values from the metadata (specifically state dicts layers)
        :return: boolean value of match (or not)\n
        note: prep with conditional `if entry.startswith("r'")`
        """
        if layer_element == "":
            raise ValueError("The value to compare from the inspected file cannot be an empty string.")
        elif type(layer_element) == str and layer_element.startswith("r'"):
            # Regex conversion
            expression = (layer_element
                          .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
                          .replace(".", r"\.")    # Escape literal dots with '\.'
                          .strip("r'")            # Strip the 'r' and quotes from the string
                          )
            print(expression)
            regex_entry = re.compile(expression)
            return bool(regex_entry.search(block_pattern))
        else:
            if type(layer_element) == str and type(block_pattern) == str:
                return block_pattern.lower() in layer_element.lower()
            elif layer_element is not None and block_pattern is not None:
                return layer_element == block_pattern
            else:
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
