# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


import os
import sys
import unittest
from unittest.mock import patch

from nnll.model_detect.layer_filter import LayerFilter
from nnll.model_detect.key_trail import KeyTrail


class TestLayerFilter(unittest.TestCase):
    def setUp(self):
        self.layer_filter = LayerFilter()

    @patch("nnll.model_detect.key_trail.KeyTrail.pull_key_names")
    def test_modelspec_bundle(self, mock_pull_key_names):
        # Mock return values for pull_key_names
        mock_pull_key_names.side_effect = ["modelspec", "unet"]

        filter_cascade = {
            "layer_type": {
                "modelspec": {"blocks": "value1"},
            },
            "category": {"unet": {"blocks": "value2"}},
        }

        model_header = {"value1": "x", "value2": "data"}
        tensors = None

        expected_result = {"layer_type": "modelspec", "category": "unet", "model": "unknown"}

        result = self.layer_filter.reference_walk_conductor(filter_cascade, model_header, tensors)
        print(result)
        self.assertEqual(result, expected_result)

    @patch("nnll.model_detect.key_trail.KeyTrail.pull_key_names")
    def test_diffusers_model(self, mock_pull_key_names):
        # Mock return value for pull_key_names
        mock_pull_key_names.side_effect = ["diffusers", "unet", "sdxl-base"]

        filter_cascade = {
            "layer_type": {"modelspec": {"blocks": "value1"}, "diffusers": {"blocks": "value2"}},
            "category": {"unet": {"blocks": "valueb"}, "language": {"blocks": "valuea"}},
            "unet": {"auraflow": {"blocks": "valuex"}, "sdxl-base": {"blocks": "valuey"}, "flux1": {"blocks": "valuez"}},
        }

        model_header = {"valueb": "value1", "value2": "valuea", "valuey": "valuez"}
        tensors = None

        expected_result = {"layer_type": "diffusers", "category": "unet", "model": "sdxl-base"}
        result = self.layer_filter.reference_walk_conductor(filter_cascade, model_header, tensors)
        self.assertEqual(result, expected_result)

    @patch("nnll.model_detect.key_trail.KeyTrail.pull_key_names")
    def test_other_criteria(self, mock_pull_key_names):
        # Mock return value for pull_key_names
        mock_pull_key_names.side_effect = ["unknown", "unet", "flux-1"]

        filter_cascade = {
            "layer_type": {"modelspec": {"blocks": "value1"}, "diffusers": {"blocks": "value2"}},
            "category": {"unet": {"blocks": "valueb"}, "language": {"blocks": "valuea"}},
            "unet": {"auraflow": {"blocks": "valuex"}, "sdxl-base": {"blocks": "valuey"}, "flux1": {"blocks": "valuez"}},
        }

        model_header = {"valuez": "value1", "valueb": "value2"}
        tensors = None

        expected_result = {"layer_type": "unknown", "category": "unet", "model": "flux-1"}
        result = self.layer_filter.reference_walk_conductor(filter_cascade, model_header, tensors)
        self.assertEqual(result, expected_result)

    @patch("nnll.model_detect.key_trail.KeyTrail.pull_key_names")
    def test_empty_bundle_data(self, mock_pull_key_names):
        # Mock return value for pull_key_names to be None
        mock_pull_key_names.side_effect = ["modelspec", "unknown", "unknown"]

        filter_cascade = {
            "layer_type": {"modelspec": {"blocks": "value1"}, "diffusers": {"blocks": "value2"}},
            "category": {"unet": {"blocks": "valueb"}, "language": {"blocks": "valuea"}},
            "unet": {"auraflow": {"blocks": "valuex"}, "sdxl-base": {"blocks": "valuey"}, "flux1": {"blocks": "valuez"}},
        }

        model_header = {"value1": "foo", "key2": "value2", "bar": "baz"}
        tensors = None

        expected_result = {"layer_type": "modelspec", "category": "unknown", "model": "unknown"}

        result = self.layer_filter.reference_walk_conductor(filter_cascade, model_header, tensors)
        self.assertEqual(result, expected_result)

    @patch("nnll.model_detect.key_trail.KeyTrail.pull_key_names")
    def test_mixed_criteria(self, mock_pull_key_names):
        # Mock return values for pull_key_names
        mock_pull_key_names.side_effect = [["modelspec"], ["unet", "language"], ["sdxl-base"], ["clip-g"]]

        filter_cascade = {
            "layer_type": {"modelspec": {"blocks": "value1"}, "diffusers": {"blocks": "value2"}},
            "category": {"unet": {"blocks": "valueb"}, "language": {"blocks": "valuea"}},
            "unet": {"auraflow": {"blocks": "valuey", "shapes": [1024, 2048], "tensors": 50}, "sdxl-base": {"blocks": "valuey", "shapes": [640, 320], "tensors": 300}, "flux1": {"blocks": "valuez", "shapes": [768, 768], "tensors": 20}},
            "language": {"clip-g": {"blocks": "valuea", "shapes": [640, 320]}, "clip-h": {"blocks": "valuex", "shapes": [640, 320]}},
        }
        model_header = {
            "valuez": {"shape": [640, 320]},  # no result
            "value1": {"shape": [640, 320]},  # establish as modelspec
            "valueb": {"shape": [640, 320]},  # establish as unet
            "valuey": {"shape": [640, 320]},  # establish as sdxl base
            "valuea": {"shape": [640, 320]},  # clip g
        }
        tensors = 300

        expected_result = {"layer_type": "modelspec", "category": "bundle", "component_name": "sdxl-base clip-g", "component_type": "unet language", "model": "sdxl-base"}
        result = self.layer_filter.reference_walk_conductor(filter_cascade, model_header, tensors)
        print(result)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
