#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s

import re
import os
import hashlib
import sys


class ExtractAndMatchMetadata:

    def extract_tensor_data(self, source_data_item: dict, id_values: dict) -> dict:
        """
        Extracts shape and key data from the source meta data and put them into id_values\n
        This would extract whatever additional information is needed when a match is found.\n
        :param layer_element: `dict` Values from the metadata (specifically state dicts layers)
        :param id_values: `dict` Collection of identifiable attributes extracted from the source item
        :return: `dict` Tensor and tensor shape attribute details from the source item
        """
        TENSOR_TOLERANCE = 4e-2  # ? : Move to a config file?
        search_items = ["dtype", "shape"]
        for field_name in search_items:
            if (field_value := source_data_item.get(field_name)) is not None:
                if isinstance(field_value, list):
                    field_value = str(field_value)  # Convert shape list to string
                existing_values = id_values.get(field_name, "").split()

                if field_value not in existing_values:  # Add the new value only if it's not already present
                    existing_values.append(field_value)
                    id_values[field_name] = " ".join(existing_values)

        return {
            "tensors": id_values.get("tensors", 0),
            'shape': id_values.get('shape', None)
        }

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
