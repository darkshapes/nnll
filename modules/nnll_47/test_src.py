#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

from modules.nnll_47.src import parse_pulled_keys
import unittest

class TestParsePulledKeys(unittest.TestCase):
    def test_parse_with_values(self):
        self.assertEqual(
            parse_pulled_keys(
                {'layer': 'layer_name_a'},
                ['category_type1'],
                ['model_arch_1']),
            {'layer': 'layer_name_a',
            'component_type': 'category_type1',
            'model_type': 'category_type1',
             'component_name': 'model_arch_1',
             'architecture': 'model_arch_1'})

    def test_parse_with_none(self):
        self.assertEqual(
            parse_pulled_keys({}, [None], [None]),
            {'component_type': 'unknown', 'model_type': 'unknown', 'component_name': 'unknown',
             'architecture': 'unknown'}
        )

if __name__ == '__main__':
    unittest.main()