# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import hashlib
import os
import unittest
from unittest.mock import MagicMock

from nnll.model_detect.layer_pattern import ExtractAndMatchMetadata
from nnll.model_detect.value_compare import ValueComparison


class TestCompareValues(unittest.TestCase):
    def setUp(self):
        self.extract = ExtractAndMatchMetadata()
        self.handle_values = ValueComparison()
        # Mocking the match_pattern_and_regex method for controlled tests
        self.extract.is_pattern_in_layer = MagicMock()
        self.attributes = {"tensors": 1}
        test_file_path = os.path.dirname(os.path.abspath(__file__))
        test_file_name = "test.txt"
        self.test_file_path_named = os.path.join(test_file_path, test_file_name)
        with open(self.test_file_path_named, "wb") as f:
            f.write(b"Hello, World!")

    def test_empty_pattern_details(self):
        pattern_details = {}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_block_pattern(self):
        pattern_details = {"blocks": "layer2"}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = True
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_non_matching_block_pattern(self):
        pattern_details = {"blocks": ["^conv.*"]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = False
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_block_list(self):
        pattern_details = {"blocks": ["layer1"]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, True]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_matching_shapes(self):
        pattern_details = {"blocks": ["layer"], "shapes": [3, 256, 256]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, True]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_non_matching_shapes(self):
        pattern_details = {"blocks": [r"^laye*1"], "shapes": [[3, 128, 128]]}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        self.extract.is_pattern_in_layer.side_effect = [True, False]
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_matching_tensors(self):
        pattern_details = {"blocks": ["layer1"], "tensors": 1}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, self.attributes)
        self.assertTrue(result)

    def test_non_matching_tensors(self):
        pattern_details = {"blocks": ["^layer.*"], "tensors": 2}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, self.attributes)
        self.assertFalse(result)

    def test_missing_tensors(self):
        pattern_details = {"blocks": ["^layer.*"], "tensors": 2}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}}
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertFalse(result)

    def test_ignore_missing_tensors(self):
        pattern_details = {"blocks": ["yer."], "tensors": 2}
        unpacked_metadata = {"layer.1": {"shape": [3, 256, 256]}}
        pattern_details.pop("tensors")
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata)
        self.assertTrue(result)

    def test_matching_hash(self):
        test_file_path = os.path.dirname(os.path.abspath(__file__))
        test_file_name = "test.txt"
        test_file_path_named = os.path.join(test_file_path, test_file_name)
        attribute = {"hash": hashlib.sha256(b"Hello, World!").hexdigest(), "file_path_named": test_file_path_named}
        with open(test_file_path_named, "wb") as f:
            f.write(b"Hello, World!")
        pattern_details = {"blocks": "layer2"} | attribute
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = True
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, attribute)
        self.assertTrue(result)

    def test_matching_hash_v2(self):
        attribute = {"file_path_named": self.test_file_path_named}
        pattern_details = {"blocks": "layer2", "hash": hashlib.sha256(b"Hello, World!").hexdigest()}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = True
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, attribute)
        self.assertTrue(result)

    def test_matching_file_size(self):
        attribute = {"file_size": 13}
        pattern_details = {"blocks": "layer2", "file_size": (os.path.getsize(self.test_file_path_named))}
        unpacked_metadata = {"layer1": {"shape": [3, 256, 256]}, "layer2": {"shape": [1, 128, 128]}}
        self.extract.is_pattern_in_layer.return_value = True
        result = self.handle_values.check_model_identity(pattern_details, unpacked_metadata, attribute)
        self.assertTrue(result)

    def tearDown(self):
        try:
            os.remove(self.test_file_path_named)
        except OSError:
            pass


if __name__ == "__main__":
    test = TestCompareValues()
    test.setUp()
    test.test_matching_block_pattern()
    test.tearDown()
