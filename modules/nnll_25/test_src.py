### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
from unittest import TestCase, mock
import pytest
import hashlib
from unittest.mock import mock_open

from modules.nnll_25.src import ExtractAndMatchMetadata

testdata_00 = [
    ("Hello 123 World", "r'd+'", True),
    ("Example.com", "r'.com'", True),
    ("Test Data 456", "r'Test d+ata'", False),  # This should be False because 'd+' is not replaced correctly
    ("123.456", "r'123.d+'", True),
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
        result = test_module.is_pattern_in_layer(reference_data, source_item_data)
        assert result == expected

    @classmethod
    def test_empty_regex(cls):
        test_module = ExtractAndMatchMetadata()
        reference_data = ["", "Hello World"]
        source_item_data = ""
        for each in reference_data:
            result = test_module.is_pattern_in_layer(each, source_item_data)
            assert result == False
