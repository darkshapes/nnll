
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_25.src import ExtractAndMatchMetadata


def find_value_path(reference_map: dict, source_item_data: dict, corresponding_values: dict = {}) -> list | None:
    """
    Find path in target nested dictionary where values match `source_item_data`.
    :param reference_data: `str` A regex pattern from known identifiers
    :param source_item_data: `str` Values from the metadata (specifically state dicts layers)
    :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
    """
    def recursive_search(nested_map: dict, current_path: list = []) -> list | None:
        run_instance_process = ExtractAndMatchMetadata()
        for k, v in nested_map.items():
            new_path = current_path + [k]

            if isinstance(v, dict):  # Check if we've reached the deepest level, and whether it matches source_item_data
                for layer, tensor_data in source_item_data.items():
                    for item, match in v.items():
                        if isinstance(match, list):
                            for each in match:
                                if run_instance_process.match_pattern_and_regex(each, layer):
                                    return new_path  # Found a match
                        else:
                            try:
                                model_dict = { "blocks": layer, "shapes": tensor_data["shape"] } | corresponding_values
                            except TypeError as errorlog:
                                """logger.errorlog things here"""
                                model_dict = source_item_data | corresponding_values
                            if all(run_instance_process.match_pattern_and_regex(v.get(key), value) for key, value in model_dict.items()):
                                return new_path if not isinstance(new_path, list) else next(iter(new_path))  # Found a match
                result = recursive_search(v, new_path)  # Recurse into deeper levels
                if result is not None:
                    return result  # Return path once found
            else:
                continue  # Skip non-dict values

        return None  # No matching subtree found

    if all(reference_map.get(key) == value for key, value in source_item_data.items()):  # Check for top-level direct match
        return list(source_item_data.keys())

    return recursive_search(reference_map)
