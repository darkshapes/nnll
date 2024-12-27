
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


from collections import OrderedDict


def key_trail(nested_dictionary : dict) -> OrderedDict:
    """
    Recursive lambda function to show sequence of keys from a nested dictionary, preserving key order
    :param nested_dictionary: `dict` A nested dictionary to track
    :returns: `OrderedDict` The series of keys in the dictionary
    """
    return OrderedDict(
    (k, None) if not isinstance(v, dict) else (k, key_trail(v))
    for k, v in nested_dictionary.items()
)

json_data = {
    "value_to_match": {"subsequent_value_to_match_a": "label_a"},
    "alternative_match": {"subsequent_value_to_match_b": "label_b"}
}
ordered_keys = key_trail(json_data)

print(ordered_keys)
