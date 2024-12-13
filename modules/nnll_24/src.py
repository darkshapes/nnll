
import os
import sys
from typing import Generator

# sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from modules.nnll_25.src import ExtractAndMatchMetadata


def compare_values(nested_filter: dict, model_header: str, tensor_count=None):
    run_instance_process = ExtractAndMatchMetadata()
    for layer_element, tensor_data in model_header.items():  # < - get the information necessary to match
        if (len(nested_filter)) > 1:  # only the json entries for specific models have more than 1 key
            model_dict = { "blocks": layer_element, "shapes": tensor_data["shape"] }  # create a structurally-mirrored dict from the unknown file data
            if tensor_count is not None:
                model_dict.set_default("tensors", tensor_count)
            else:
                if nested_filter.get("tensors", 0) != 0:
                    nested_filter.pop("tensors")
            if all(run_instance_process.match_pattern_and_regex(nested_filter.get(key), value) for key, value in model_dict.items()):
                return True
        else:
            for _, match in nested_filter.items():
                if not isinstance(match, list):
                    raise TypeError("Comparison values received in non-`list` format. Verify model reference category/layer formatting.")
                else:
                    for block_pattern in match:
                        if run_instance_process.match_pattern_and_regex(block_pattern, layer_element):
                            return True
    return False


def find_value_path(filter_cascade: dict, model_header: dict, tensor_count: dict = None) -> list | None:
    """
    Find path in target nested dictionary where values match `model_header`.
    :param filter_cascade: `dict` A dictionary of regex patterns and criteria known to identify models
    :param model_header: `dict` Values from the unknown file metadata (specifically state dicts layers)
    :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
    """
    def recursive_search(nested_filter: dict, current_path: list = []) -> Generator | list | None:
        for k, v in nested_filter.items():
            new_path = current_path + [k]

            if isinstance(v, dict):  # Check if we've reached the deepest level, and whether it matches model_header
                if compare_values(v, model_header, tensor_count) == True:
                    return new_path  # Found a match

                result = recursive_search(v, new_path)  # Recurse into deeper levels
                if result is not None:
                    return result  # Return path once found
            else:
                continue  # Skip non-dict values

        return None  # No matching subtree found

    if all(filter_cascade.get(key) == value for key, value in model_header.items()):  # Check for top-level direct match
        # return list(model_header.keys()) #return all keys
        return list(model_header.keys())[-1]

    return recursive_search(filter_cascade)
