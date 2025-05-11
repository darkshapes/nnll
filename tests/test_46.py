### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, Mock

from nnll_46 import IdConductor
from nnll_24 import KeyTrail


class TestIDConductor(unittest.TestCase):
    def setUp(self):
        self.processor = IdConductor()
        self.key_trail = KeyTrail()
        self.modelspec_layer = {"layer_type": "modelspec"}
        self.non_modelspec_layer = {"layer_type": "non-modelspec"}
        self.pattern_reference = {
            "layer_type": {"pulled_type1": "data", "pulled_type2": "data"},
            "category": {
                "lora": {"blocks": ["lora"]},
                "tae": {"blocks": "decoder.layers"},
                "unet": {"blocks": ["diffusion_model", "model.diffusion", "img_", ".img"]},
                "language": {"blocks": ["SelfAttention", "self_attention", "self_attn"]},
                "vae": {"blocks": ["decoder.mid_", "decoder.up"]},
            },
        }
        self.attributes = {}
        self.unpacked_metadata = {"diffusion_model": "{'shape': [1024]}", "self_attention": "{'shape': [2048]}"}

    def test_process_file_metadata_modelspec(self):
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            self.attributes["tensors"] = 1105

            self.pulled_keys = self.processor.identify_category_type(self.modelspec_layer, self.pattern_reference, self.unpacked_metadata, self.attributes)

            mock_pull_key_names.assert_any_call(self.pattern_reference["category"], self.unpacked_metadata, self.attributes)
            for category in list(self.pattern_reference["category"])[3:]:
                mock_pull_key_names.assert_any_call(self.pattern_reference["category"][category], self.unpacked_metadata)

    def test_file_metadata_not_modelspec(self):
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            self.attributes["tensors"] = 1105
            self.pulled_keys = self.processor.identify_category_type(self.non_modelspec_layer, self.pattern_reference, self.unpacked_metadata, self.attributes)

            mock_pull_key_names.assert_called_once_with(self.pattern_reference["category"], self.unpacked_metadata, self.attributes)

    def test_identify_layer_type(self):
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            mock_pull_key_names.return_value = ["pulled_type2"]
            id_keys = self.processor.identify_layer_type(self.pattern_reference, self.unpacked_metadata, 5)
            expected_id_keys = {"layer_type": ["pulled_type2"]}
            self.assertEqual(id_keys, expected_id_keys)

            self.unpacked_metadata = {"metadata_key": "metadata_value"}

    def test_single_model_type(self):
        self.pattern_reference = {"type1": "pattern1", "type2": "pattern2"}
        self.unpacked_metadata = {"metadata_key": "metadata_value"}
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            # Test with a single model type and tensor count.
            mock_pull_key_names.return_value = "key1"
            self.attributes["tensors"] = 5
            result = self.processor.identify_model("type1", self.pattern_reference, self.unpacked_metadata, self.attributes)
            self.assertEqual(result, ["key1"])
            mock_pull_key_names.assert_called_once_with("pattern1", {"metadata_key": "metadata_value"}, {"tensors": 5})

    def test_multiple_model_types(self):
        self.pattern_reference = {"type1": "pattern1", "type2": "pattern2"}
        self.unpacked_metadata = {"metadata_key": "metadata_value"}
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            mock_pull_key_names.side_effect = ["key1", "key2"]
            self.attributes["tensors"] = 5
            result = self.processor.identify_model(["type1", "type2"], self.pattern_reference, self.unpacked_metadata, self.attributes)
            self.assertEqual(result, ["key1", "key2"])
            calls = [(("pattern1", {"metadata_key": "metadata_value"}, {"tensors": 5}), {}), (("pattern2", {"metadata_key": "metadata_value"}), {})]
            mock_pull_key_names.assert_has_calls(calls)

    def test_invalid_model_type(self):
        with self.assertRaises(KeyError):
            self.processor.identify_model("invalid_type", self.pattern_reference, self.unpacked_metadata)

    def test_empty_model_types(self):
        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            mock_pull_key_names.return_value = ["key1"]
            result = self.processor.identify_model([], self.pattern_reference, self.unpacked_metadata)
            self.assertEqual(result, [])
            mock_pull_key_names.assert_not_called()

    def test_none_attributes(self):
        self.pattern_reference = {"type1": "pattern1", "type2": "pattern2"}
        self.unpacked_metadata = {"metadata_key": "metadata_value"}

        with patch("nnll_24.KeyTrail.pull_key_names") as mock_pull_key_names:
            mock_pull_key_names.side_effect = ["key1"]
            result = self.processor.identify_model("type1", self.pattern_reference, self.unpacked_metadata)
            self.assertEqual(result, ["key1"])
            mock_pull_key_names.assert_called_once_with("pattern1", {"metadata_key": "metadata_value"}, None)


if __name__ == "__main__":
    unittest.main()
