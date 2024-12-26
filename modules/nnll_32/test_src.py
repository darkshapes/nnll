#
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

# Assuming get_model_header is in a module named model_loader
from modules.nnll_32.src import get_model_header

class TestGetModelHeader(unittest.TestCase):

    def setUp(self):
        self.file_path = 'test_file'
        self.safetensors_extension = '.safetensors'
        self.gguf_extension = '.gguf'
        self.pickletensor_extension = '.pt'
        self.invalid_extension = '.txt'

        # Mock file metadata
        self.mock_model_header = {'key': 'value'}
        self.mock_disk_size = 1024

    @patch('modules.nnll_32.src.Path')
    @patch('modules.nnll_32.src.metadata_from_safetensors')
    def test_safetensors_file(self, mock_metadata_from_safetensors, mock_path):
        mock_path.return_value.suffix.lower.return_value = self.safetensors_extension
        with patch('os.path.getsize', return_value=self.mock_disk_size), \
             patch('os.path.basename', return_value='test_file' + self.safetensors_extension):
            mock_metadata_from_safetensors.return_value = self.mock_model_header

            result = get_model_header(self.file_path)
            expected_result = (self.mock_model_header, self.mock_disk_size, 'test_file' + self.safetensors_extension, self.safetensors_extension)

            self.assertEqual(result, expected_result)

    @patch('modules.nnll_32.src.Path')
    @patch('modules.nnll_32.src.metadata_from_gguf')
    def test_gguf_file(self, mock_metadata_from_gguf, mock_path):
        mock_path.return_value.suffix.lower.return_value = self.gguf_extension
        with patch('os.path.getsize', return_value=self.mock_disk_size), \
             patch('os.path.basename', return_value='test_file' + self.gguf_extension):
            mock_metadata_from_gguf.return_value = self.mock_model_header

            result = get_model_header(self.file_path)
            expected_result = (self.mock_model_header, self.mock_disk_size, 'test_file' + self.gguf_extension, self.gguf_extension)

            self.assertEqual(result, expected_result)

    @patch('modules.nnll_32.src.Path')
    @patch('modules.nnll_32.src.load_pickletensor_metadata_from_model')
    def test_pickletensor_file(self, mock_load_pickletensor_metadata_from_model, mock_path):
        mock_path.return_value.suffix.lower.return_value = self.pickletensor_extension
        with patch('os.path.getsize', return_value=self.mock_disk_size), \
             patch('os.path.basename', return_value='test_file' + self.pickletensor_extension):
            mock_load_pickletensor_metadata_from_model.return_value = self.mock_model_header

            result = get_model_header(self.file_path)
            expected_result = (self.mock_model_header, self.mock_disk_size, 'test_file' + self.pickletensor_extension, self.pickletensor_extension)

            self.assertEqual(result, expected_result)

    @patch('modules.nnll_32.src.Path')
    def test_invalid_extension(self, mock_path):
        mock_path.return_value.suffix.lower.return_value = self.invalid_extension

        result = get_model_header(self.file_path)
        self.assertIsNone(result)

    def test_empty_or_none_extension(self):
        # Test for empty extension
        with patch('modules.nnll_32.src.Path') as mock_path:
            mock_path.return_value.suffix.lower.return_value = ''
            result = get_model_header(self.file_path)
            self.assertIsNone(result)

        # Test for None extension
        with patch('modules.nnll_32.src.Path') as mock_path:
            mock_path.return_value.suffix.lower.return_value = None
            result = get_model_header(self.file_path)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()