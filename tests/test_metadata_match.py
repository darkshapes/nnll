### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
from unittest import TestCase, mock
import pytest
import hashlib

from nnll.integrity.hashing import compute_hash_for


class TestExtractAndMatchMetadata(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create a temporary test file for permission and I/O error tests
        cls.test_file_name = "test.txt"
        with open(cls.test_file_name, "wb") as f:
            f.write(b"Hello, World!")

    def test_valid_file(self):
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        assert compute_hash_for(self.test_file_name) == expected_hash

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            compute_hash_for("nonexistent_file.txt")

    @mock.patch("builtins.open", side_effect=PermissionError)
    def test_permission_error(self, mock_open):
        with pytest.raises(PermissionError) as exc_info:
            compute_hash_for(self.test_file_name)
        self.assertEqual(type(exc_info.value), PermissionError)

    @mock.patch("builtins.open", side_effect=IOError)
    def test_io_error(self, mock_open):
        with pytest.raises(OSError) as exc_info:
            compute_hash_for("test.txt")

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove(cls.test_file_name)
        except OSError:
            pass
