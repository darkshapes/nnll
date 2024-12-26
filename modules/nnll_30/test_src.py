
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import json
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from modules.nnll_30.src import write_json_file, read_json_file


class TestFileOperations(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to store the test files
        self.temp_dir = TemporaryDirectory()
        self.file_name = "test_data.json"
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)
        self.test_data = {
            "key1": "value1",
            "key2": 69,  # nice
            "key3": [1, 2, 3],
        }

    def test_write_and_read_json_file(self):
        # Write data to a JSON file
        write_json_file(self.temp_dir.name, self.file_name, self.test_data)

        # Read data back from the JSON file
        read_data = read_json_file(self.file_path)

        # Assert that the written and read data are the same
        self.assertEqual(read_data, self.test_data)

    def test_read_nonexistent_file(self):
        # Test reading a non-existent file should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            read_json_file("non_existent_file.json")

    # @patch('builtins.print')
    # def test_invalid_json_format(self, mock_print):
    #     # Create a mock JSON file with invalid content
    #     content = '{"tensor_1": {"shape": [2, 3, 4]}/}'
    #     with open(self.file_path, 'w') as f:
    #         f.write(content)

    #     read_json_file(self.file_path)
    #     expected_output = "json.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 34 (char 33)"
    #     mock_print.assert_called_once_with(expected_output)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()
