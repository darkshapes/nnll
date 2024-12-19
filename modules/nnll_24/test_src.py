
import pytest
import os
import sys
import unittest
from unittest.mock import MagicMock

from modules.nnll_24.src import ValueComparisons
from modules.nnll_25.src import ExtractAndMatchMetadata


class TestCompareValues(unittest.TestCase):
    def setUp(self):
        self.extract = ExtractAndMatchMetadata()
        self.handle_values = ValueComparisons()
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

    def test_find_value_path(self):
        nested_filter = {
            'x': {'blocks': 'c1d2', 'shapes': [256]},
            'z': {'blocks': "c2d2", 'shapes': [512]},
            'y': {'blocks': 'c.d', 'shapes': [256]}}

        # Test matching path
        model_header = {'c.d': {'shape': [256]}}
        assert self.handle_values.find_value_path(nested_filter, model_header) == ['y']

        # Test no match found
        model_header_no_match = {'e': 3, 'f': 4}
        assert self.handle_values.find_value_path(nested_filter, model_header_no_match) is None

        # Test deeper nested structure
        nested_filter = {
            'level1': {
                'level2': {
                    'level3': {
                        'blocks': 'c.d',
                        'shapes': [256]
                    }
                },
                'another': {'skip': {}}
            }
        }
        assert self.handle_values.find_value_path(nested_filter, model_header) == ['level1', 'level2', 'level3']

        # Test with empty dict
        nested_filter_empty = {}
        assert self.handle_values.find_value_path(nested_filter_empty, model_header) is None

        # Test matching at the top level
        model_header = {'c.d': {'shape': [256]}}
        assert self.handle_values.find_value_path(nested_filter, model_header) == ['level1', 'level2', 'level3']

        # test block, shape, tensor match combined
        nested_filter = {
            'w': {
                'blocks': "c.e",
                'shapes': [256],
                'tensors': 244
            },
            'y': {
                'blocks': "c.d",
                'shapes': [
                    256
                ],
                'tensors': 244
            },
            'z': {
                'blocks': "c.d",
                'tensors': 244
            },
            'x': {
                'blocks': 'c.d',
            }
        }
        assert self.handle_values.find_value_path(nested_filter, model_header, tensor_count=244) == ['y']

        # block and shape only

        model_header = {
            "c.d": {
                "shape": [
                    256
                ]
            }
        }
        assert self.handle_values.find_value_path(nested_filter, model_header) == ['y']

        # empty shape
        model_header = {
            "c.d": {
                "shape": [
                ]
            }
        }
        assert self.handle_values.find_value_path(nested_filter, model_header, tensor_count=243) == ['x']

        # no shape, only tensor!
        model_header = {
            "c.d": {}
        }
        assert self.handle_values.find_value_path(nested_filter, model_header, tensor_count=244) == ['z']


if __name__ == '__main__':
    unittest.main()
