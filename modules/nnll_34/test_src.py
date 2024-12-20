# import unittest
# import re

# from modules.nnll_34.src import get_last_num_from_filename

# class TestGetLastNumFromFilename(unittest.TestCase):

#     def test_basic_match(self):
#         filename = "model_02of3.safetensors"
#         search_value = "of"
#         self.assertEqual(get_last_num_from_filename(filename, search_value), 2)

#     def test_no_match(self):
#         filename = "model_02of5.safetensors"
#         search_value = "xyz"
#         with self.assertLogs(level='WARNING') as log:
#             result = get_last_num_from_filename(filename, search_value)
#             self.assertIsNone(result)
#             self.assertIn("File not named with search_term", log.output[0])

#     def test_single_digit_match(self):
#         filename = "model_1of3.safetensors"
#         search_value = "of"
#         self.assertEqual(get_last_num_from_filename(filename, search_value), 1)

#     def test_leading_non_digit(self):
#         filename = "model_02of3.safetensors"
#         search_value = "of"
#         self.assertEqual(get_last_num_from_filename(filename, search_value), 2)

#     def test_multiple_digits(self):
#         filename = "model_99of100.safetensors"
#         search_value = "of"
#         self.assertIsNone(get_last_num_from_filename(filename, search_value))  # Current implementation only supports up to 2 digits

#     def test_no_digits_before_search_value(self):
#         filename = "model_of3.safetensors"
#         search_value = "of"
#         self.assertIsNone(get_last_num_from_filename(filename, search_value))

#     def test_special_characters(self):
#         filename = "model_02-of-3.safetensors"
#         search_value = "-of-"
#         self.assertEqual(get_last_num_from_filename(filename, search_value), 2)

#     def test_case_insensitive_search(self):
#         filename = "Model_02Of3.SAFETENSORS"
#         search_value = "oF"
#         self.assertEqual(get_last_num_from_filename(filename, search_value), 2)

# if __name__ == '__main__':
#     unittest.main()