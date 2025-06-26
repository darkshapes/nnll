### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import hashlib
import unittest

from nnll.model_detect.key_trail import KeyTrail


class TestKeyTrail(unittest.TestCase):
    def setUp(self):
        self.attributes = {"tensors": 244}
        test_file_path = os.path.dirname(os.path.abspath(__file__))
        test_file_name = "test.txt"
        self.test_file_path_named = os.path.join(test_file_path, test_file_name)
        with open(self.test_file_path_named, "wb") as f:
            f.write(b"Software design is my passion.")
        return KeyTrail()

    def test_pull_key_names_match(self):
        key_trail = self.setUp()
        nested_filter = {"x": {"blocks": "c1d2", "shapes": [256]}, "z": {"blocks": "c2d2", "shapes": [512]}, "y": {"blocks": "c.d", "shapes": [256]}}
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, "y")

    def test_pull_key_names_no_match(self):
        key_trail = self.setUp()
        nested_filter = {"x": {"blocks": "c1d2", "shapes": [256]}, "z": {"blocks": "c2d2", "shapes": [512]}, "y": {"blocks": "c.d", "shapes": [256]}}
        model_header_no_match = {"e": 3, "f": 4}
        self.assertIsNone(key_trail.pull_key_names(nested_filter, model_header_no_match))

    def test_pull_key_names_deeper_nested(self):
        key_trail = self.setUp()
        nested_filter = {"level1": {"level2": {"level3": {"blocks": "c.d", "shapes": [256]}}, "another": {"skip": {}}}}
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, "level3")

    def test_pull_key_names_empty_dict(self):
        key_trail = self.setUp()
        nested_filter_empty = {}
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter_empty, model_header)
        self.assertIsNone(output)

    def test_pull_key_names_top_level_match(self):
        key_trail = self.setUp()
        nested_filter = {"level1": {"level2": {"level3": {"blocks": "c.d", "shapes": [256]}}}}
        model_header = {"c.d": {"shape": [256]}}
        self.assertEqual(key_trail.pull_key_names(nested_filter, model_header), "level3")

    def test_pull_key_names_combined_match(self):
        key_trail = self.setUp()
        nested_filter = {"w": {"blocks": "c.e", "shapes": [256], "tensors": 244}, "y": {"blocks": "c.d", "shapes": [256], "tensors": 244}, "z": {"blocks": "c.d", "tensors": 244}, "x": {"blocks": "c.d"}}
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, self.attributes)
        self.assertEqual(output, "y")

    def test_pull_key_names_block_shape_only(self):
        key_trail = self.setUp()
        nested_filter = {"w": {"blocks": "c.e", "shapes": [256], "tensors": 244}, "y": {"blocks": "c.d", "shapes": [256], "tensors": 244}, "z": {"blocks": "c.d", "tensors": 244}, "x": {"blocks": "c.d", "shapes": [256]}}
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header)
        self.assertEqual(output, "y")

    def test_pull_key_names_empty_shape_empty_tensor(self):
        key_trail = self.setUp()
        nested_filter = {
            "w": {"blocks": "c.e", "shapes": [256], "tensors": 244},
            "y": {"blocks": "c.d", "shapes": [256], "tensors": 244},
            "z": {"blocks": "c.d", "tensors": 244},  ####
            "x": {"blocks": "c.d"},
        }
        model_header = {"c.d": {}}
        self.attributes["tensors"] = 0
        output = key_trail.pull_key_names(nested_filter, model_header, self.attributes)  # Pass 0
        self.assertEqual(output, "x")

    def test_pull_key_names_no_shape_only_tensor(self):
        key_trail = self.setUp()
        nested_filter = {"w": {"blocks": "c.e", "shapes": [256], "tensors": 244}, "y": {"blocks": "c.d", "shapes": [256], "tensors": 244}, "z": {"blocks": "c.d", "tensors": 244}, "x": {"blocks": "c.d"}}
        model_header = {"c.d": {}}
        output = key_trail.pull_key_names(nested_filter, model_header, self.attributes)  # Passes 244
        self.assertEqual(output, "z")

    def test_pull_key_names_combined_match_block_list(self):
        key_trail = self.setUp()
        nested_filter = {
            "x": {"blocks": "c.e"},
            "z": {"blocks": "c.e", "tensors": 244},
            "y": {"blocks": ["c.l", "c.d"], "tensors": 244},  ### because c.d is in it
            "w": {"blocks": ["c.e", "c.d"], "shapes": [256], "tensors": 244},
        }
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, self.attributes)
        self.assertEqual(output, "y")

    def test_pull_key_names_combined_match_tensor_list(self):
        key_trail = self.setUp()
        nested_filter = {
            "x": {"blocks": "c.e"},
            "y": {"blocks": ["c.l", "c.d"], "tensors": [24, 4]},
            "z": {"blocks": "c.d", "shapes": [25], "tensors": [222, 244]},
            "w": {"blocks": ["c.e", "c.d"], "shapes": [256], "tensors": [222, 244]},
        }  ###
        model_header = {"c.d": {"shape": [256]}}
        output = key_trail.pull_key_names(nested_filter, model_header, self.attributes)
        self.assertEqual(output, "w")

    def test_pull_key_names_hash_match(self):
        key_trail = self.setUp()
        nested_filter = {
            "x": {
                "blocks": "c1d2",
                "shapes": [256],
            },
            "z": {"blocks": "c2d2", "shapes": [512], "hash": "6f79c1397cb9ce1dac363722dbe70147aee0ccca75e28338f8482fe515891399"},
            "y": {"blocks": "c.d", "shapes": [256], "hash": "ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f"},
            "w": {"blocks": "c.d", "shapes": [256], "hash": hashlib.sha256(b"Software design is my passion.").hexdigest()},
        }
        model_header = {"c.d": {"shape": [256]}}
        attributes = {}  # Clear any tensor info out
        attributes["file_path_named"] = self.test_file_path_named
        output = key_trail.pull_key_names(nested_filter, model_header, attributes)
        self.assertEqual(output, "w")

    def test_pull_key_names_file_size_match(self):
        key_trail = self.setUp()
        nested_filter = {
            "w": {"blocks": "c.d", "shapes": [256], "file_size": "0"},
            "x": {"blocks": "c.d", "shapes": [256], "file_size": "8192"},
            "z": {"blocks": "c.d", "shapes": [256], "file_size": 30},
            "y": {"blocks": "c.d", "shapes": [256], "file_size": "1024"},
        }
        model_header = {"c.d": {"shape": [256]}}
        attributes = {}  # Clear any tensor info out
        print(os.path.getsize(self.test_file_path_named))
        attributes["file_size"] = os.path.getsize(self.test_file_path_named)
        output = key_trail.pull_key_names(nested_filter, model_header, attributes)
        self.assertEqual(output, "z")

    def tearDown(self):
        try:
            os.remove(self.test_file_path_named)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
