

import os
from unittest import TestCase, mock
import pytest
import hashlib
from unittest.mock import patch, mock_open, MagicMock
from functools import reduce

import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_25.src import ExtractAndMatchMetadata


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
    ("", "", False),  # Empty reference data and source item data
    ("Hello World", "", False),  # Empty source item data
    ("", "r'\\d+'", False),  # Empty reference data with regex pattern
]

testdata_05 = [
    ("Hello World", r"Hello World", False),  # Invalid regex pattern (unescaped quotes)
]

combined_testdata = testdata_00 + testdata_01 + testdata_02 + testdata_03 + testdata_04


class TestRegex:
    @pytest.mark.parametrize("reference_data,source_item_data,expected", combined_testdata)
    def test_valid_regex(self, reference_data, source_item_data, expected):
        test_module = ExtractAndMatchMetadata()
        result = test_module.match_pattern_and_regex(reference_data, source_item_data)
        assert result == expected


testdata_00 = [     # Test basic functionality
    ({"dtype": "float32", "shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"dtype": "int64", "shape": [5, 5]}, {"dtype": "float32"}, {"tensors": 0, "shape": "[5, 5]", "dtype": "float32 int64"}),
]

testdata_01 = [    # Test with existing values in id_values
    ({"dtype": "float32", "shape": [10, 20]}, {"dtype": "float32", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5] [10, 20]", "dtype": "float32"}),
    ({"dtype": "int64", "shape": [10, 20]}, {"dtype": "float32 int64", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5] [10, 20]", "dtype": "float32 int64"}),
]

testdata_02 = [    # Test with list and string conversion
    ({"dtype": "float32"}, {}, {"tensors": 0, "shape": None}),
    ({"shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({}, {"dtype": "float32", "shape": "[5, 5]"}, {"tensors": 0, "shape": "[5, 5]", "dtype": "float32"}),
]
testdata_03 = [    # Test with missing fields in source_data_item
    ({"shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"shape": [1, 2, 3]}, {"shape": "[10, 20]"}, {"tensors": 0, "shape": "[10, 20] [1, 2, 3]"}),
]
testdata_04 = [  # Test with empty source_data_item and id_values
    ({}, {}, {"tensors": 0, "shape": None}),
]
testdata_05 = [
    ({"dtype": "float32", "shape": 10}, {}, {"tensors": 0, "shape": "10"}),
]
testdata_06 = [
    ({"dtype": None, "shape": [10, 20]}, {}, {"tensors": 0, "shape": "[10, 20]"}),
    ({"dtype": "float32", "shape": None}, {}, {"tensors": 0, "shape": None}),
]

combined_testdata = testdata_00 + testdata_01 + testdata_02 + testdata_03 + testdata_04 + testdata_05 + testdata_06


class TensorTest:

    @pytest.mark.parametrize("source_data_item, id_values, expected", )
    def test_basic_functionality(source_data_item, id_values, expected):
        test_module = ExtractAndMatchMetadata()
        assert test_module.extract_tensor_data(source_data_item, id_values) == expected
