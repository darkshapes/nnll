
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest

from modules.nnll_24.src import KeyTrail

class TestKeyTrail(unittest.TestCase):

    def setUp(self):
        return KeyTrail()

    def test_pull_key_names_match(self):
        key_trail = self.setUp()
        nested_filter = {'x': {'blocks': 'c1d2', 'shapes': [256]}, 'z': {'blocks': "c2d2", 'shapes': [512]}, 'y': {'blocks': 'c.d', 'shapes': [256]}}
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, 'y')

    def test_pull_key_names_no_match(self):
        key_trail = self.setUp()
        nested_filter = {'x': {'blocks': 'c1d2', 'shapes': [256]}, 'z': {'blocks': "c2d2", 'shapes': [512]}, 'y': {'blocks': 'c.d', 'shapes': [256]}}
        model_header_no_match = {'e': 3, 'f': 4}
        self.assertIsNone(key_trail.pull_key_names(nested_filter, model_header_no_match))

    def test_pull_key_names_deeper_nested(self):
        key_trail = self.setUp()
        nested_filter = {
            'level1': {
                'level2': {
                    'level3': {'blocks': 'c.d', 'shapes': [256]}
                },
                'another': {'skip': {}}
            }
        }
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, 'level3')

    def test_pull_key_names_empty_dict(self):
        key_trail = self.setUp()
        nested_filter_empty = {}
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter_empty, model_header)
        self.assertIsNone(output)

    def test_pull_key_names_top_level_match(self):
        key_trail = self.setUp()
        nested_filter = {
            'level1': {
                'level2': {
                    'level3': {'blocks': 'c.d', 'shapes': [256]}
                }
            }
        }
        model_header = {'c.d': {'shape': [256]}}
        self.assertEqual(key_trail.pull_key_names(nested_filter, model_header), 'level3')

    def test_pull_key_names_combined_match(self):
        key_trail = self.setUp()
        nested_filter = {
            'w': {'blocks': "c.e", 'shapes': [256], 'tensors': 244},
            'y': {'blocks': "c.d", 'shapes': [256], 'tensors': 244},
            'z': {'blocks': "c.d", 'tensors': 244},
            'x': {'blocks': 'c.d'}
        }
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, tensor_count=244)
        self.assertEqual(output, 'y')

    def test_pull_key_names_block_shape_only(self):
        key_trail = self.setUp()
        nested_filter = {
            'w': {'blocks': "c.e", 'shapes': [256], 'tensors': 244},
            'y': {'blocks': "c.d", 'shapes': [256], 'tensors': 244},
            'z': {'blocks': "c.d", 'tensors': 244},
            'x': {'blocks': 'c.d', 'shapes': [256]}
        }
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, 'y')

    def test_pull_key_names_empty_shape(self):
        key_trail = self.setUp()
        nested_filter = {
            'w': {'blocks': "c.e", 'shapes': [256], 'tensors': 244},
            'y': {'blocks': "c.d", 'shapes': [256], 'tensors': 244},
            'z': {'blocks': "c.d", 'tensors': 244},
            'x': {'blocks': 'c.d'}, ####
        }
        model_header = {'c.d': {}}
        output = key_trail.pull_key_names(nested_filter, model_header, tensor_count=244)
        self.assertEqual(output, 'z')

    def test_pull_key_names_no_shape_only_tensor(self):
        key_trail = self.setUp()
        nested_filter = {
            'w': {'blocks': "c.e", 'shapes': [256], 'tensors': 244},
            'y': {'blocks': "c.d", 'shapes': [256], 'tensors': 244},
            'z': {'blocks': "c.d", 'tensors': 244},
            'x': {'blocks': 'c.d'}
        }
        model_header = {'c.d': {}}
        output = key_trail.pull_key_names(nested_filter, model_header, tensor_count=244)
        self.assertEqual(output, 'z')

    def test_pull_key_names_combined_match_block_list(self):
        key_trail = self.setUp()
        nested_filter = {
            'x': {'blocks': 'c.e'},
            'z': {'blocks': "c.e", 'tensors': 244},
            'y': {'blocks': ["c.l","c.d"],'tensors': 244}, ### because c.d is in it
            'w': {'blocks': ["c.e","c.d"], 'shapes': [256], 'tensors': 244},

        }
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, tensor_count=244)
        self.assertEqual(output, 'y')

    def test_pull_key_names_combined_match_tensor_list(self):
        key_trail = self.setUp()
        nested_filter = {
            'x': {'blocks': 'c.e'},
            'y': {'blocks': ["c.l","c.d"],'tensors': [24,4]},
            'z': {'blocks': "c.d", 'shapes': [25],'tensors': [222,244]},
            'w': {'blocks': ["c.e","c.d"], 'shapes': [256], 'tensors': [222,244]},
        } ###
        model_header = {'c.d': {'shape': [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, tensor_count=244)
        self.assertEqual(output, 'w')
if __name__ == '__main__':
    unittest.main()
