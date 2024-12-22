#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import unittest

from modules.nnll_00.src import get_matching_layer

class TestJsonComparison(unittest.TestCase):

    def setUp(self):
        # Example dict for testing
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
        # Test should return the first key ("header")
        result = get_matching_layer(self.json_data)
        self.assertEqual(result, "header")

    def test_empty_dict(self):
        # Empty dict should return 0
        result = get_matching_layer({})
        self.assertEqual(result, 0)
