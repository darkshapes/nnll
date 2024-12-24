#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import os
from unittest import TestCase, mock
import pytest
import hashlib
from unittest.mock import mock_open

from modules.nnll_25.src import ExtractAndMatchMetadata


class AttributeFunctionTests(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Create a temporary test file for permission and I/O error tests
        cls.test_module = ExtractAndMatchMetadata()
        cls.test_file_name = "test.txt"
        with open(cls.test_file_name, 'wb') as f:
            f.write(b"Hello, World!")

    def test_valid_file(cls):
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        assert cls.test_module.compute_file_hash(cls.test_file_name) == expected_hash

    def test_nonexistent_file(cls):
        with pytest.raises(FileNotFoundError):
            cls.test_module.compute_file_hash('nonexistent_file.txt')

    @mock.patch('builtins.open', side_effect=PermissionError)
    def test_permission_error(cls, mock_open):
        with pytest.raises(PermissionError) as exc_info:
            cls.test_module.compute_file_hash(cls.test_file_name)
        cls.assertEqual(type(exc_info.value), PermissionError)

    @mock.patch('builtins.open', side_effect=IOError)
    def test_io_error(cls, mock_open):
        with pytest.raises(OSError) as exc_info:
            cls.test_module.compute_file_hash("n.txt")
        assert "File 'n.txt' does not exist." in str(exc_info.value)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove(cls.test_file_name)
        except OSError:
            pass


testdata_00 = [
    ("Hello 123 World", "r'd+'", True),
    ("Example.com", "r'.com'", True),
    ("Test Data 456", "r'Test d+ata'", False),  # This should be False because 'd+' is not replaced correctly
    ("123.456", "r'123.d+'", True)
]

testdata_01 = [
    ("Hello World", "r'y'", False),  # Invalid regex pattern
    ("Example.com", "r'[a-z]+.com'", True),  # No match found
]

testdata_02 = [
    ("Hello World", "r'World'", True),
    ("Example.com", "com'", False),  # No exact match
    ("Test Data", "r'Test Data'", True),
]
testdata_03 = [
    ("123.456.789", "r'd+.d+.d+'", True),
    ("Special @#$%^&*", "Special @#$%^&*", True),
    ("Escaped \\ and \\", "Escaped \\ and \\", True),
]

testdata_04 = [
    ("", "r'\\d+'", False),  # Empty reference data with regex pattern
]

testdata_05 = [
    ("Hello World", r"Hello World", False),  # Invalid regex pattern (unescaped quotes)
]

combined_testdata = testdata_00 + testdata_01 + testdata_02 + testdata_03


class TestRegex:
    @pytest.mark.parametrize("reference_data,source_item_data,expected", combined_testdata)
    def test_valid_regex(self, reference_data, source_item_data, expected):
        test_module = ExtractAndMatchMetadata()
        result = test_module.match_pattern_and_regex(reference_data, source_item_data)
        assert result == expected

    @classmethod
    def test_empty_regex(cls):
        test_module = ExtractAndMatchMetadata()
        reference_data = ["", "Hello World"]
        source_item_data = ""
        expected = ValueError("The value to compare from the inspected file cannot be an empty string.")  # Empty reference data and source item data
        with pytest.raises(ValueError) as exc_info:
            for each in reference_data:
                result = test_module.match_pattern_and_regex(each, source_item_data)
                assert expected == exc_info

