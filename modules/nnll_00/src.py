
import unittest


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
