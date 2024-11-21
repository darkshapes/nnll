

from collections import OrderedDict

get_keys_ordered = lambda d: OrderedDict(
    (k, None) if not isinstance(v, dict) else (k, get_keys_ordered(v))
    for k, v in d.items()
)# Lambda to create OrderedDict of nested dictionaries, preserving key order

json_data = {
    "value_to_match": { "subsequent_value_to_match_a": "label_a" },
    "alternative_match": { "subsequent_value_to_match_b": "label_b" }
}
ordered_keys = get_keys_ordered(json_data)

print(ordered_keys)




# import unittest
# from collections import OrderedDict

# # Updated Python code that compares the JSON data to a dict
# def get_all_keys(json_dict):
#     # Lambda to recursively get all keys from nested dicts, preserving order using OrderedDict
#     get_keys = lambda d: OrderedDict((k, None) for k, v in d.items())
#     return get_keys(json_dict)

# # Define a test case using unittest
# class TestJsonComparison(unittest.TestCase):

#     def test_all_keys(self):
#         # Set up example JSON-like dict for testing
#         json_data = {
#                     "value_to_match": "label_a",
#                     "alternative_match": "label_b"
#                     }


#         # Test that the lambda correctly retrieves all nested keys in order using OrderedDict
#         all_keys = get_all_keys(json_data)
#         expected_keys = OrderedDict([
#             ('header', None),
#             ('category', None),
#             ('value_to_match', None),
#             ('subsequent_value_to_match_a', None),
#             ('alternative_match', None),
#             ('subsequent_value_to_match_b', None)
#         ])

#         print(list(all_keys).keys())  # Print the sequence of keys
#         self.assertEqual(list(all_keys.keys()), list(expected_keys.keys()))  # Compare the key sequences

# if __name__ == '__main__':
#     unittest.main(argv=['first-arg-is-ignored'], exit=False)
