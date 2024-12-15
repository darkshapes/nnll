
import os
import sys
from typing import Generator

# sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_25.src import ExtractAndMatchMetadata


def compare_values(nested_filter: dict, model_header: dict, tensor_count: int | None = None) -> bool:
    """
    Iterate through model header in order to run a check on model_header based on nested_filter
    :param nested_filter: `dict` A dictionary of regex patterns and criteria known to identify models
    :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
    :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
    :return: `bool` Whether or not the values from the model header and tensor_count were found inside nested_filter
    """
    extract = ExtractAndMatchMetadata()

    # Iterate through the filter, run the scan
    # Determine what needs to be done with the data, then execute

    for layer, tensor_data in model_header.items():  # repeat for every incoming layer
        if isinstance(nested_filter.get("blocks", 0), list) == True:
            for pattern in nested_filter["blocks"]:  # repeat for each list item
                match = extract.match_pattern_and_regex(pattern, layer)
                if match == True:
                    break

        else:
            match = extract.match_pattern_and_regex(nested_filter.get("blocks"), layer)

        if match == True:
            model_dict = {}
            # Fetch any remaining items necessary to compare
            if nested_filter.get("tensors", None) is None and nested_filter.get("shapes", None) is None:
                return True
            if nested_filter.get("shapes", None) is not None:
                model_dict.setdefault("shapes", tensor_data.get("shape", 0))
            if tensor_count is not None and nested_filter.get("tensors", None) is not None:
                model_dict.setdefault("tensors", tensor_count)
            elif tensor_count is None and nested_filter.get("tensors", None) is not None:
                # if 'tensor_count was not supplied, ignore it
                nested_filter.pop("tensors")

            # Do a quick match of all values at once
            if all(extract.match_pattern_and_regex(nested_filter.get(key), value) for key, value in model_dict.items()):
                return True
    return False


def find_value_path(filter_cascade: dict, model_header: dict, tensor_count: int | None = None) -> list | None:
    """
    Recurse through `filter_cascade` dictionary and return a key if it matches k/v pair details in `model_header`.
    :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
    :param model_header: `dict` Values from the unknown file metadata (specifically state dict layers)
    :param tensor_count: `dict` Optional numer of model layers in the unknown model file as an integer (None will bypass necessity of a match)
    :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
    """
    def recursive_search(nested_filter: dict, current_path: list = []) -> Generator | list | None:
        for id, attributes in nested_filter.items():
            new_path = current_path + [id]  # k becomes our check data, new path the identifying attribute

            if isinstance(attributes, dict):  # Check if we've reached the deepest level, and whether it matches model_header
                if compare_values(attributes, model_header, tensor_count) == True:
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
