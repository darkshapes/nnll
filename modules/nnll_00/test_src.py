
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import unittest

from modules.nnll_00.src import deepest_key_of

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
        result = deepest_key_of(self.json_data)
        self.assertEqual(result, "header")

    def test_empty_dict(self):
        # Empty dict should return 0
        result = deepest_key_of({})
        self.assertEqual(result, 0)
