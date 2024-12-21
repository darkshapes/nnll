
import unittest
import re

from modules.nnll_35.src import get_count_from_filename

class TestGetLastNumFromFilename(unittest.TestCase):

    def test_basic_match(self):
        filename = "model_02of03.safetensors"
        search_value = "of"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

    def test_no_match(self):
        filename = "model_02of5.safetensors"
        search_value = "xyz"
        result = get_count_from_filename(filename, search_value)
        self.assertIsNone(result)
        self.assertRaises(ValueError)

    def test_single_digit_match(self):
        filename = "model_1of3.safetensors"
        search_value = "of"
        self.assertEqual(get_count_from_filename(filename, search_value), 1)

    def test_leading_non_digit(self):
        filename = "model_02of3.safetensors"
        search_value = "of"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

    def test_multiple_digits(self):
        filename = "model_100of100.safetensors"
        search_value = "of"
        self.assertEqual(get_count_from_filename(filename, search_value),00)  # Current implementation only supports up to 2 digits

    def test_no_digits_before_search_value(self):
        filename = "modelof3.safetensors"
        search_value = "of"
        result = get_count_from_filename(filename, search_value)
        self.assertIsNone(result)

    def test_special_characters(self):
        filename = "model_02-of3.safetensors"
        search_value = "-of"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

    def test_case_insensitive_search(self):
        filename = "Model_02OF3.Safetensors"
        search_value = "oF"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

    def test_search_value_at_start(self):
        filename = "02modelof3.safetensors"
        search_value = "model"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

    def test_search_value_at_end(self):
        filename = "model_02.safetensors"
        search_value = ".safetensors"
        self.assertEqual(get_count_from_filename(filename, search_value), 2)

if __name__ == '__main__':
    unittest.main()