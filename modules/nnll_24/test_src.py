#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import pytest
import os
import sys
import unittest

from modules.nnll_24.src import ValuePath

class TestCompareValues(unittest.TestCase):
    def setUp(self):
        self.handle_values = ValuePath()
        # Mocking the match_pattern_and_regex method for controlled tests

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
