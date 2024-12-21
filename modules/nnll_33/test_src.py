#// SPDX-License-Identifier: MIT
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
        self.extract.match_pattern_and_regex = MagicMock()

    def test_empty_nested_filter(self):
        nested_filter = {}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertFalse(result)

    def test_matching_block_pattern(self):
        nested_filter = {"blocks": "layer2"}
        model_header = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.match_pattern_and_regex.return_value = True
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertTrue(result)

    def test_non_matching_block_pattern(self):
        nested_filter = {"blocks": ["^conv.*"]}
        model_header = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.match_pattern_and_regex.return_value = False
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertFalse(result)

    def test_matching_block_list(self):
        nested_filter = {"blocks": ["layer1"]}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.match_pattern_and_regex.side_effect = [True, True]
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertTrue(result)

    def test_matching_shapes(self):
        nested_filter = {"blocks": ["layer"], "shapes": [3, 256, 256]}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.match_pattern_and_regex.side_effect = [True, True]
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertTrue(result)

    def test_non_matching_shapes(self):
        nested_filter = {"blocks": ["^layer.*"], "shapes": [[3, 128, 128]]}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.match_pattern_and_regex.side_effect = [True, False]
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertFalse(result)

    def test_matching_tensors(self):
        nested_filter = {"blocks": ["layer"], "tensors": 1}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.compare_values(nested_filter, model_header, tensor_count=1)
        self.assertTrue(result)

    def test_non_matching_tensors(self):
        nested_filter = {"blocks": ["^layer.*"], "tensors": 2}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.compare_values(nested_filter, model_header, tensor_count=1)
        self.assertFalse(result)

    def test_missing_tensors(self):
        nested_filter = {"blocks": ["^layer.*"], "tensors": 2}
        model_header = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertFalse(result)

    def test_ignore_missing_tensors(self):
        nested_filter = {"blocks": ["yer."], "tensors": 2}
        model_header = {"layer.1": {"shape": [3, 256, 256]}}
        nested_filter.pop("tensors")
        result = self.handle_values.compare_values(nested_filter, model_header)
        self.assertTrue(result)
