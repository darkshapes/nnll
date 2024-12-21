

from collections import OrderedDict


def get_ordered_keys(data : dict) -> OrderedDict:
    """
    Recursive lambda function to show sequence of keys from a nested dictionary, preserving key order
    :param data: `dict` A nested dictionary to track
    :returns: `OrderedDict` The series of keys in the dictionary
    """
    return OrderedDict(
    (k, None) if not isinstance(v, dict) else (k, get_ordered_keys(v))
    for k, v in data.items()
)

json_data = {
    "value_to_match": {"subsequent_value_to_match_a": "label_a"},
    "alternative_match": {"subsequent_value_to_match_b": "label_b"}
}
ordered_keys = get_ordered_keys(json_data)

print(ordered_keys)
