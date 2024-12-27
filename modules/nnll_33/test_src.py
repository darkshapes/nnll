
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest
from unittest.mock import MagicMock

from modules.nnll_33.src import ValueComparison
from modules.nnll_25.src import ExtractAndMatchMetadata

class TestCompareValues(unittest.TestCase):

    def setUp(self):
        self.extract = ExtractAndMatchMetadata()
        self.handle_values = ValueComparison()
        # Mocking the match_pattern_and_regex method for controlled tests
        self.extract.is_pattern_in_layer = MagicMock()

    def test_empty_pattern_details(self):
        pattern_details = {}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_block_pattern(self):
        pattern_details = {"blocks": "layer2"}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = True
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_non_matching_block_pattern(self):
        pattern_details = {"blocks": ["^conv.*"]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = False
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_block_list(self):
        pattern_details = {"blocks": ["layer1"]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, True]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_matching_shapes(self):
        pattern_details = {"blocks": ["layer"], "shapes": [3, 256, 256]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, True]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_non_matching_shapes(self):
        pattern_details = {"blocks": [r'^laye*1'], "shapes": [[3, 128, 128]]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, False]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_tensors(self):
        pattern_details = {"blocks": ["layer1"], "tensors": 1}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, tensor_count=1)
        self.assertTrue(result)

    def test_non_matching_tensors(self):
        pattern_details = {"blocks": ["^layer.*"], "tensors": 2}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, tensor_count=1)
        self.assertFalse(result)

    def test_missing_tensors(self):
        pattern_details = {"blocks": ["^layer.*"], "tensors": 2}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_ignore_missing_tensors(self):
        pattern_details = {"blocks": ["yer."], "tensors": 2}
        unpacked_metadata = {"layer.1": {"shape": [3, 256, 256]}}
        pattern_details.pop("tensors")
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

if __name__== "__main__":
    test = TestCompareValues()
    test.setUp()
    test.test_matching_block_pattern()