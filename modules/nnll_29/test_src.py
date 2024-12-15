
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0])) ))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0]), "modules") ))

from nnll_29.src import BlockScanner


class TestBlockScanner(unittest.TestCase):

    def setUp(self):
        self.block_scanner = BlockScanner()

    @patch('nnll_24.src.find_value_path')
    def test_compvis_bundle(self, mock_find_value_path):
        # Mock return values for find_value_path
        mock_find_value_path.side_effect = ['compvis', 'unet']

        filter_cascade = {
            'layer_type': {
                'compvis': {
                    'blocks': 'value1'
                },
            },
            'category': {
                'unet': {
                    'blocks': 'value2'
                }
            }
        }

        model_header = {'value1': 'x', 'value2': 'data'}
        tensor_count = None

        expected_result = {'layer_type': 'compvis', 'category': 'unet', 'model': 'unknown'}

        result = self.block_scanner.filter_metadata(filter_cascade, model_header, tensor_count)
        print(result)
        self.assertEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_diffusers_model(self, mock_find_value_path):
        # Mock return value for find_value_path
        mock_find_value_path.return_value = 'sdxl-base'

        filter_cascade = {
            'layer_type': {
                'compvis': {
                    'blocks': 'value1'
                },
                'diffusers': {
                    'blocks': 'value2'
                }
            },
            'category': {
                'unet': {
                    'blocks': 'valueb'
                },
                'language': {
                    'blocks': 'valuea'
                }
            },
            'unet': {
                'auraflow': { "blocks": 'valuex'},
                'sdxl-base': { "blocks": 'valuey'},
                'flux1': { "blocks": 'valuez'}
            }
        }

        model_header = {'valueb': 'value1', 'value2': 'valuea', 'valuey': 'valuez' }
        tensor_count = {}

        expected_result = {'layer_type': 'diffusers', 'category': 'unet', 'model': 'sdxl-base'}
        result = self.block_scanner.filter_metadata(filter_cascade, model_header, tensor_count)
        print(result)
        print(expected_result)
        self.assertEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_other_criteria(self, mock_find_value_path):
        # Mock return value for find_value_path
        mock_find_value_path.return_value = 'flux1'

        filter_cascade = {
            'layer_type': {
                'compvis': {
                    'blocks': 'value1'
                },
                'diffusers': {
                    'blocks': 'value2'
                }
            },
            'category': {
                'unet': {
                    'blocks': 'valueb'
                },
                'language': {
                    'blocks': 'valuea'
                }
            },
            'unet': {
                'auraflow': { "blocks": 'valuex'},
                'sdxl-base': { "blocks": 'valuey'},
                'flux1': { "blocks": 'valuez'}
            }
        }

        model_header = {'valuez': 'value1', 'valueb': 'value2'}
        tensor_count = {}

        expected_result = {'layer_type': 'unknown', 'category': 'unet', 'model': 'flux1'}
        result = self.block_scanner.filter_metadata(filter_cascade, model_header, tensor_count)
        print(result)
        self.assertEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_empty_bundle_data(self, mock_find_value_path):
        # Mock return value for find_value_path to be None
        mock_find_value_path.return_value = None

        filter_cascade = {
            'layer_type': {
                'compvis': {
                    'blocks': 'value1'
                },
                'diffusers': {
                    'blocks': 'value2'
                }
            },
            'category': {
                'unet': {
                    'blocks': 'valueb'
                },
                'language': {
                    'blocks': 'valuea'
                }
            },
            'unet': {
                'auraflow': { "blocks": 'valuex'},
                'sdxl-base': { "blocks": 'valuey'},
                'flux1': { "blocks": 'valuez'}
            }
        }

        model_header = {'value1': 'foo', 'key2': 'value2', 'bar': 'baz'}
        tensor_count = {}

        expected_result = {

            "layer_type": "compvis",
            "category": "unknown",
            "model": "unknown"
        }

        result = self.block_scanner.filter_metadata(filter_cascade, model_header, tensor_count)
        self.assertEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_mixed_criteria(self, mock_find_value_path):
        # Mock return values for find_value_path
        mock_find_value_path.side_effect = [['compvis'], ['unet', 'language'], ['sdxl', 'clip-g']]

        filter_cascade = {
            'layer_type': {
                'compvis': {
                    'blocks': 'value1'
                },
                'diffusers': {
                    'blocks': 'value2'
                }
            },
            'category': {
                'unet': {
                    'blocks': 'valueb'
                },
                'language': {
                    'blocks': 'valuea'
                }
            },
            'unet': {
                'auraflow': { "blocks": 'valuey', "shapes": [1024, 2048], "tensors": 50},
                'sdxl-base': { "blocks": 'valuey', "shapes": [640, 320], "tensors": 300},
                'flux1': { "blocks": 'valuez', "shapes": [768, 768], "tensors": 20}
            },
            'language': {
                "clip-g": { "blocks": 'valuea', "shapes": [640, 320]},
                "clip-h": { "blocks": 'valuex', "shapes": [640, 320]}
            }
        }
        model_header = {
            'valuez': {'shape': [640, 320]},
            'value1': {'shape': [640, 320]},
            'valueb': {'shape': [640, 320]},
            'valuey': {'shape': [640, 320]},
            'valuea': {'shape': [640, 320]}
        }
        tensor_count = 300

        expected_result = {
            'layer_type': 'compvis',
            'category': 'bundle',
            'component_name': 'sdxl-base clip-g',
            'component_type': 'unet language',
            'model': 'sdxl-base'
        }
        result = self.block_scanner.filter_metadata(filter_cascade, model_header, tensor_count)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
