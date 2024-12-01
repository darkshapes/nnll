

from collections import OrderedDict


def get_keys_ordered(d): return OrderedDict(
    (k, None) if not isinstance(v, dict) else (k, get_keys_ordered(v))
    for k, v in d.items()
)  # Lambda to create OrderedDict of nested dictionaries, preserving key order


json_data = {
    "value_to_match": {"subsequent_value_to_match_a": "label_a"},
    "alternative_match": {"subsequent_value_to_match_b": "label_b"}
}
ordered_keys = get_keys_ordered(json_data)

print(ordered_keys)
