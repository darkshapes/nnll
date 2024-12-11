
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from modules.nnll_25.src import ExtractAndMatchMetadata


def compare_values(nested_ref: dict, source_item_data: str, tensor_count):
    run_instance_process = ExtractAndMatchMetadata()
    for layer_element, tensor_data in source_item_data.items():
        if (len(nested_ref)) > 1:  # the dicts are shaped to only find one value
            model_dict = { "blocks": layer_element, "shapes": tensor_data["shape"] } | (tensor_count if tensor_count is not None else {})
            if all(run_instance_process.match_pattern_and_regex(nested_ref.get(key), value) for key, value in model_dict.items()):
                return True
        else:
            for _, match in nested_ref.items():
                if not isinstance(match, list):
                    raise TypeError("Comparison values received in non-`list` format. Verify model reference category/layer formatting.")
                else:
                    for block_pattern in match:
                        if run_instance_process.match_pattern_and_regex(block_pattern, layer_element):
                            return True
    return False


def find_value_path(reference_map: dict, source_item_data: dict, tensor_count: dict = None) -> list | None:
    """
    Find path in target nested dictionary where values match `source_item_data`.
    :param reference_data: `str` A regex pattern from known identifiers
    :param source_item_data: `str` Values from the metadata (specifically state dicts layers)
    :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
    """
    def recursive_search(nested_map: dict, current_path: list = []) -> list | None:
        for k, v in nested_map.items():
            new_path = current_path + [k]

            if isinstance(v, dict):  # Check if we've reached the deepest level, and whether it matches source_item_data
                if compare_values(v, source_item_data, tensor_count) == True:
                    return new_path  # Found a match

                result = recursive_search(v, new_path)  # Recurse into deeper levels
                if result is not None:
                    return result  # Return path once found
            else:
                continue  # Skip non-dict values

        return None  # No matching subtree found

    if all(reference_map.get(key) == value for key, value in source_item_data.items()):  # Check for top-level direct match
        # return list(source_item_data.keys()) #return all keys
        return list(source_item_data.keys())[-1]

    return recursive_search(reference_map)
