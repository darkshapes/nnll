

#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import os
from unittest import TestCase, mock
import pytest
import hashlib

from modules.nnll_44.src import compute_file_hash

class TestExtractAndMatchMetadata(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Create a temporary test file for permission and I/O error tests
        cls.test_file_name = "test.txt"
        with open(cls.test_file_name, 'wb') as f:
            f.write(b"Hello, World!")

    def test_valid_file(cls):
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        assert compute_file_hash(cls.test_file_name) == expected_hash

    def test_nonexistent_file(cls):
        with pytest.raises(FileNotFoundError):
            compute_file_hash('nonexistent_file.txt')

    @mock.patch('builtins.open', side_effect=PermissionError)
    def test_permission_error(cls, mock_open):
        with pytest.raises(PermissionError) as exc_info:
            compute_file_hash(cls.test_file_name)
        cls.assertEqual(type(exc_info.value), PermissionError)

    @mock.patch('builtins.open', side_effect=IOError)
    def test_io_error(cls, mock_open):
        with pytest.raises(OSError) as exc_info:
            compute_file_hash("n.txt")
        assert "File 'n.txt' does not exist." in str(exc_info.value)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove(cls.test_file_name)
        except OSError:
            pass