# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


import os
import unittest
from tempfile import TemporaryDirectory
from nnll.metadata.json_io import write_json_file, read_json_file


class TestFileOperations(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory to store the test files"""
        self.temp_dir = TemporaryDirectory()
        self.file_name = "test_data.json"
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)
        self.test_data = {
            "key1": "value1",
            "key2": 69,  # nice
            "key3": [1, 2, 3],
        }

    def test_write_and_read_json_file(self):
        """Write data to a JSON file, Read data back from the JSON file,Assert that the written and read data are the same"""
        write_json_file(self.temp_dir.name, self.file_name, self.test_data)
        read_data = read_json_file(self.file_path)
        self.assertEqual(read_data, self.test_data)

    def test_read_nonexistent_file(self):
        """Test reading a non-existent file should raise FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            read_json_file("non_existent_file.json")

    def tearDown(self):
        """Clean up the temporary directory"""
        self.temp_dir.cleanup()


if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
