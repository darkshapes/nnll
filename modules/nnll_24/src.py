

def find_value_path(reference_map: dict, file_tags: dict) -> list | None:
    """
    Find path in target nested dictionary where values match `file_tags`.\n
    :param reference_map: `dict` The taget dictionary structure to search within.
    :param file_tags: `dict` Key-value pairs to search for.
    :return: `list` The path of keys through the target `dict` leading to a matching subtree, or None if no match is found.
    """
    def recursive_search(nested_dict: dict, current_path: list = []) -> list | None:
        for k, v in nested_dict.items():
            new_path = current_path + [k]

            if isinstance(v, dict):  # Check if we've reached the deepest level, and whether it matches file_tags
                if all(v.get(key) == value for key, value in file_tags.items()):
                    return new_path  # Found a match

                result = recursive_search(v, new_path)  # Recurse into deeper levels
                if result is not None:
                    return result  # Return path once found
            else:
                continue  # Skip non-dict values

        return None  # No matching subtree found

    if all(reference_map.get(key) == value for key, value in file_tags.items()):  # Check for top-level direct match
        return list(file_tags.keys())

    return recursive_search(reference_map)
