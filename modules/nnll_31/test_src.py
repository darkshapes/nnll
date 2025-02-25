### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
import os
import json
from unittest.mock import patch, mock_open

from modules.nnll_31.src import count_tensors_and_extract_shape


class TestCountTensorsAndExtractShape(unittest.TestCase):
    def setUp(self):
        self.file_path = "test.json"
        self.pattern = "double_blocks.0.img_mlp.2.weight.quant_map"

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    @patch("builtins.print")
    def test_pattern_found_valid_json(self, mock_print):
        # Create a mock JSON file with valid content
        content = {"double_blocks.0.img_attn.norm.query_norm.scale": {"dtype": "F32", "shape": [128], "data_offsets": [512, 1024]}, "double_blocks.0.img_mlp.2.weight.quant_map": {"dtype": "F32", "shape": [16], "data_offsets": [7079104, 7079168]}}

        with open(self.file_path, "w", encoding="UTF-8") as f:
            json.dump(content, f)

        count_tensors_and_extract_shape(self.pattern, self.file_path)

        expected_output = (self.file_path, {"shapes": "[16]", "tensors": 2})
        mock_print.assert_called_once_with(*expected_output)

    @patch("builtins.print")
    def test_partial_pattern(self, mock_print):
        # Create a mock JSON file with valid content
        content = {
            "double_blocks.0.img_attn.norm.query_norm.scale": {"dtype": "F32", "shape": [16], "data_offsets": [512, 1024]},
            "norm_out.linear.weight": {"dtype": "BF16", "shape": [6144, 3072], "data_offsets": [12288, 37761024]},
            "double_blocks.0.img_mlp.2.weight.quant_map": {"dtype": "F32", "shape": [16], "data_offsets": [7079104, 7079168]},
        }

        with open(self.file_path, "w", encoding="UTF-8") as f:
            json.dump(content, f)

        count_tensors_and_extract_shape(self.pattern, self.file_path)

        expected_output = (self.file_path, {"shapes": "[16]", "tensors": 3})
        mock_print.assert_called_once_with(*expected_output)

    @patch("builtins.print")
    def test_pattern_not_found(self, mock_print):
        # Create a mock JSON file with valid content but no matching pattern
        content = {"add_embedding.linear_2.bias": {"dtype": "F16", "shape": [1280], "data_offsets": [7211520, 7214080]}}
        with open(self.file_path, "w", encoding="UTF-8") as f:
            json.dump(content, f)

        count_tensors_and_extract_shape(self.pattern, self.file_path)

        # No output expected since the pattern is not found
        mock_print.assert_not_called()

    # @patch("builtins.print")
    def test_io_error(self):
        # Simulate an I/O error
        with patch("modules.nnll_30.src.read_json_file", side_effect=IOError("File not found")):
            count_tensors_and_extract_shape(self.pattern, self.file_path)
            # expected_output = "Error reading file test.json: [Errno 2] No such file or directory: 'test.json'"
            # assert_called_once_with(expected_output)


if __name__ == "__main__":
    unittest.main()
