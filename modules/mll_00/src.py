"""

iterator_operated_on = [map(lambda each: [for each] do_operate, [using_this_iterable])]
single_value_operated_on = from functools import reduce; reduce(lambda each: [for each] do_operate, [start at]0))
iterator_filtered_as_true = [filter(lambda each: x modulo 2 == 0 (no remainder), [iterable])]


>>> doubled_numbers = list(map(lambda x: x * 2, numbers))

inline functions invoking immediately with the value of x.
>>> doubled_numbers = [(lambda x: x * 2)(x) for x in numbers]
>>> even_numbers = [x for x in numbers if (lambda x: x % 2 == 0)(x)]
>>> person_info = [(lambda name, age: (name, age) )(name, age) for name, age in zip(names, ages)] name and age lists are combined into a dict
>>> sorted_points = sorted(points, key=lambda x: x[1]) (1 as in index of second element)

if the block name is regex, interpret the expression.
    pass the expression to a value
compare the block name one time for both types of information.

parameter_name = map(lambda stored_value: stored_value in json_file_dict)
matching_layer = next(filter(lambda layer: layer == parameter_name, state_dict), 0)
"""


import unittest

# Updated Python code that compares the JSON data to a dict
def get_matching_layer(json_file_dict):
    """`
    Test of effectiveness with `map` `lambda` function for nested `dict` traversal
    `json_file_dict` a dictionary of features from recognized models
    """
    matching_layer = next(iter(
        map(lambda stored_value: stored_value, json_file_dict.keys())
        ), 0)
    return matching_layer


# Define a test case using unittest
class TestJsonComparison(unittest.TestCase):

    def setUp(self):
        # Set up example JSON-like dict for testing
        self.json_data = {
            "header": {
                "category": {
                    "value_to_match": {
                        "subsequent_value_to_match": "label_a"
                    },
                    "alternative_match": {
                        "subsequent_value_to_match": "label_b"
                    }
                }
            }
        }

    def test_first_key(self):
        # Test with a dict that should return the first key ("header")
        result = get_matching_layer(self.json_data)
        self.assertEqual(result, "value_to_match")

    def test_empty_dict(self):
        # Test with an empty dict, should return 0
        result = get_matching_layer({})
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
