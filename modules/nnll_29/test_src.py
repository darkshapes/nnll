
import unittest
from collections import defaultdict
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0]))))

from src import BlockScanner


class TestBlockScanner(unittest.TestCase):

    def setUp(self):
        self.block_scanner = BlockScanner()
        self.layer_data = {
            "model": {
                "unet": {
                    "auraflow": {
                        "blocks": "model.double_layers.3.mlpX.c_proj.weight",
                        "shapes": [3072, 3072],
                        "tensors": 849
                    },
                    "flux": {
                        "blocks": "double_blocks.12.txt_mod.lin.weight",
                        "shapes": [18432, 3072],
                        "tensors": 780
                    },
                    "hunyuan": {
                        "blocks": "model.blocks.39.skip_norm.weight",
                        "shapes": [2816],
                        "tensors": 2166
                    }
                },
                "language": {
                    "t5-xxl": {
                        "blocks": "text_encoders.t5xxl.transformer.shared.weight",
                        "shapes": [4096]
                    },
                    "mt5-xl": {
                        "blocks": "text_encoders.mt5xl.transformer.shared.weight",
                        "shapes": [250112, 2048]
                    }
                }
            },
            "category": ["compvis"],
            "layer_type": ["unet"]
        }
        self.model_header = {
            "auraflow": {
                "blocks": "model.double_layers.3.mlpX.c_proj.weight",
                "shapes": [3072, 3072],
                "tensors": 849
            },
            "flux": {
                "blocks": "double_blocks.12.txt_mod.lin.weight",
                "shapes": [18432, 3072],
                "tensors": 780
            }
        }
        self.tensor_count = {"auraflow": 849, "flux": 780}

    @patch('nnll_24.src.find_value_path')
    def test_filter_metadata_bundle(self, mock_find_value_path):
        # Mock the return values of find_value_path for different calls
        mock_find_value_path.side_effect = [
            ["unet"],  # For model_header with tensor_count=849
            ["flux"],  # For model_header with tensor_count=780
            ["compvis"]  # For category
        ]

        expected_result = {
            "category": ["bundle"],
            "component_type": ["unet", "flux"],
            "model": ["unet", "flux"]
        }

        result = self.block_scanner.filter_metadata(self.layer_data, self.model_header, self.tensor_count)
        self.assertDictEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_filter_metadata_no_bundle(self, mock_find_value_path):
        # Mock the return values of find_value_path for different calls
        mock_find_value_path.side_effect = [
            ["unet"],  # For model_header with tensor_count=849
            ["flux"],  # For model_header with tensor_count=780
            "unknown"  # For category (no match)
        ]

        expected_result = {
            "category": "unknown",
            "model": ["unet", "flux"]
        }

        result = self.block_scanner.filter_metadata(self.layer_data, self.model_header, self.tensor_count)
        self.assertDictEqual(result, expected_result)

    @patch('nnll_24.src.find_value_path')
    def test_filter_metadata_unknown_model(self, mock_find_value_path):
        # Mock the return values of find_value_path for different calls
        mock_find_value_path.side_effect = [
            None,  # For model_header with tensor_count=849 (no match)
            None,  # For model_header with tensor_count=780 (no match)
            ["compvis"]  # For category
        ]

        expected_result = {
            "category": ["compvis"],
            "model": "unknown"
        }

        result = self.block_scanner.filter_metadata(self.layer_data, self.model_header, self.tensor_count)
        self.assertDictEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
