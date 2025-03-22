#
### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint:disable=redefined-outer-name
import unittest
from unittest.mock import patch
from unittest import mock


import pytest

from nnll_04 import ModelTool
from nnll_32 import coordinate_header_tools


@pytest.fixture(scope="session")
def mock_model_tools():
    with mock.patch("nnll_04.ModelTool", new_callable=mock.MagicMock()) as mock_tool:
        return mock_tool


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
        with patch("nnll_04.ModelTool") as mock_model_tools:
            result = coordinate_header_tools(self.file_path, ".safetensors")
            mock_model_tools.assert_called_once()

        # self.assertEqual(result, expected_result)

    def test_gguf_file(self):
        with patch("nnll_04.ModelTool") as mock_model_tools:
            result = coordinate_header_tools(self.file_path, ".gguf")
            mock_model_tools.assert_called_once()

    def test_pickletensor_file(self):
        with patch("nnll_04.ModelTool") as mock_model_tools:
            result = coordinate_header_tools(self.file_path, ".pt")
            mock_model_tools.assert_called_once()

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
