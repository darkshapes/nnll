#
### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

# Assuming get_model_header is in a module named model_loader
from nnll_04 import metadata_from_safetensors
from nnll_05 import metadata_from_gguf
from nnll_28 import metadata_from_pickletensor
from nnll_32 import coordinate_header_tools


class TestGetModelHeader(unittest.TestCase):
    def setUp(self):
        self.file_path = "test_file"
        self.safetensors_extension = ".safetensors"
        self.gguf_extension = ".gguf"
        self.pickletensor_extension = ".pt"
        self.invalid_extension = ".txt"

        # Mock file metadata
        self.mock_model_header = {"key": "value"}
        self.mock_disk_size = 1024

    def test_safetensors_file(self):
        result = coordinate_header_tools(self.file_path, ".safetensors")
        expected_result = metadata_from_safetensors

        self.assertEqual(result, expected_result)

    def test_gguf_file(self):
        result = coordinate_header_tools(self.file_path, ".gguf")
        expected_result = metadata_from_gguf

        self.assertEqual(result, expected_result)

    def test_pickletensor_file(self):
        result = coordinate_header_tools(self.file_path, ".pt")
        expected_result = metadata_from_pickletensor

        self.assertEqual(result, expected_result)

    @patch("nnll_32.Path")
    def test_invalid_extension(self, mock_path):
        mock_path.return_value.suffix.lower.return_value = self.invalid_extension

        result = coordinate_header_tools(self.file_path, self.invalid_extension)
        self.assertIsNone(result)

    def test_empty_or_none_extension(self):
        # Test for empty extension
        with patch("nnll_32.Path") as mock_path:
            mock_path.return_value.suffix.lower.return_value = ""
            result = coordinate_header_tools(self.file_path, "")
            self.assertIsNone(result)

        # Test for None extension
        with patch("nnll_32.Path") as mock_path:
            mock_path.return_value.suffix.lower.return_value = None
            result = coordinate_header_tools(self.file_path, None)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
