import unittest

from modules.nnll_00.src import get_matching_layer

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
        self.assertEqual(result, "header")

    def test_empty_dict(self):
        # Test with an empty dict, should return 0
        result = get_matching_layer({})
        self.assertEqual(result, 0)
