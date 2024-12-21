
import os
import sys
from typing import Generator

from modules.nnll_33.src import ValueComparison

class ValuePath:
    """
    Class for route tracing nested dictionary keys
    """

    @classmethod
    def find_value_path(cls, filter_cascade: dict, model_header: dict, tensor_count: int | None = None) -> list | None:
        """
        Recurse through `filter_cascade` dictionary, if an entry matches k/v pair details in `model_header`, return the parent key
        :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
        """
        compare = ValueComparison
        def recursive_search(nested_filter: dict, current_path: list = []) -> Generator | list | None:
            for id, attributes in nested_filter.items():
                new_path = current_path + [id]  # k becomes our check data, new path the identifying attribute

                if isinstance(attributes, dict):  # Check if we've reached the deepest level, and whether it matches model_header
                    if compare.compare_values(attributes, model_header, tensor_count) == True:
                        return new_path

                    result = recursive_search(attributes, new_path)  # Recurse into deeper levels
                    if result is not None:
                        return result  # Return path once found
                else:
                    continue  # Skip non-dict values

            return None  # No matching subtree found

        if all(filter_cascade.get(key) == value for key, value in model_header.items()):  # Check for top-level direct match
            return list(model_header.keys())

        return recursive_search(filter_cascade)
