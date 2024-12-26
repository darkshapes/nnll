
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import sys
from typing import Generator

from modules.nnll_33.src import ValueComparison

class KeyTrail:
    """
    Class for route tracing nested dictionary keys
    """

    @classmethod
    def pull_keys(cls, pattern_reference: dict, unpacked_metadata: dict, tensor_count: int | None = None) -> list | None:
        """
        Instantiate comparison function and run a quick check on top-level criteria, otherwise run recursion function\n
        :param pattern_reference: `dict` A dictionary of regex patterns and criteria
        :param unpacked_metadata: `dict` Values from the unknown file metadata (created for state dict layers)
        :param tensor_count: `dict` Optional number of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
        """
        compare = ValueComparison
        def sink_into(next_pattern_reference: dict, flat_key_trail: list = []) -> Generator | list | None:
            """
            Recurse through dictionary and return parent keys on boolean condition\n
            :param next_pattern_reference: `dict` A sub-key from `pattern_reference`
            :param flat_key_trail: `list` Sub-keys of `pattern_reference` leading to the current
            :return: `list` The path of keys through the target `dict` if true, or None.
            """
            for current_key, pattern_details in next_pattern_reference.items():
                current_trail = flat_key_trail + [current_key]

                if isinstance(pattern_details, dict):  # Check if we've reached the bottom
                    if compare.compare_values(pattern_details, unpacked_metadata, tensor_count) == True:
                        return current_trail

                    detected_key = sink_into(pattern_details, current_trail)  # Recurse into deeper levels
                    if detected_key is not None:
                        return detected_key  # Return path once found
                else:
                    continue  # Skip non-dict values

            return None  # No matching subtree found

        if all(pattern_reference.get(key) == value for key, value in unpacked_metadata.items()):  # In case of top-level match
            return list(unpacked_metadata.keys())

        return sink_into(pattern_reference) # Begin recursion loop
