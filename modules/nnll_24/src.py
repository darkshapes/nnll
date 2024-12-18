
import os
import sys
from typing import Generator


sys.path.append(os.path.abspath(os.path.join(os.path.pardir, "nnll", "modules")))
from nnll_25.src import ExtractAndMatchMetadata


class ValueComparisons:

    @classmethod
    def compare_values(cls, nested_filter: dict, model_header: dict, tensor_count: int | None = None) -> bool:
        """
        Iterate through model header in order to compare the nested_filter pattern against model_header\n
        If "blocks" is a list, iterate through it, checking both regex and str pattern matches.
        :param nested_filter: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `bool` Whether or not the values from the model header and tensor_count were found inside nested_filter
        """
        extract = ExtractAndMatchMetadata()

        if nested_filter is not None or nested_filter != {}:
            # Iterate through the filter, run the scan
            # Determine what needs to be done with the data, then execute
            if len(nested_filter) != 0 and nested_filter.get("blocks") is not None:
                for layer, tensor_data in model_header.items():  # repeat for every incoming layer
                    if isinstance(nested_filter["blocks"], list) == True:
                        for pattern in nested_filter["blocks"]:  # repeat for each list item
                            cls.match = extract.match_pattern_and_regex(pattern, layer)
                            if cls.match == True:
                                break

                    else:
                        cls.match = extract.match_pattern_and_regex(nested_filter["blocks"], layer)

                    if hasattr(cls, "match") :
                        if cls.match == True:
                            model_dict = {}
                            # Fetch any remaining items necessary to compare
                            if nested_filter.get("tensors", None) is None and nested_filter.get("shapes", None) is None:
                                return True
                            if nested_filter.get("shapes", None) is not None:
                                model_dict.setdefault("shapes", tensor_data.get("shape", 0))
                            if nested_filter.get("tensors", None) is not None and tensor_count is not None:
                                model_dict.setdefault("tensors", tensor_count)
                            elif tensor_count is None and nested_filter.get("tensors", None) is not None:
                                # if 'tensor_count was not supplied, ignore it
                                nested_filter.pop("tensors")

                            # Do a quick match of all values at once
                            if all(extract.match_pattern_and_regex(nested_filter.get(key), value) for key, value in model_dict.items()):
                                return True
        return False

    @classmethod
    def find_value_path(cls, filter_cascade: dict, model_header: dict, tensor_count: int | None = None) -> list | None:
        """
        Recurse through `filter_cascade` dictionary, if an entry matches k/v pair details in `model_header`, return the parent key
        :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
        :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
        :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
        :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
        """
        def recursive_search(nested_filter: dict, current_path: list = []) -> Generator | list | None:
            for id, attributes in nested_filter.items():
                new_path = current_path + [id]  # k becomes our check data, new path the identifying attribute

                if isinstance(attributes, dict):  # Check if we've reached the deepest level, and whether it matches model_header
                    if cls.compare_values(attributes, model_header, tensor_count) == True:
                        return new_path

                    result = recursive_search(attributes, new_path)  # Recurse into deeper levels
                    if result is not None:
                        return result  # Return path once found
                else:
                    continue  # Skip non-dict values

            return None  # No matching subtree found

        if all(filter_cascade.get(key) == value for key, value in model_header.items()):  # Check for top-level direct match
            # return list(model_header.keys()) #return all keys
            return list(model_header.keys())

        return recursive_search(filter_cascade)
