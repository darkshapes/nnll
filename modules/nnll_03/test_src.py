
import unittest
from unittest import TestCase
from src import find_matching_model

mock_data = {
    "neuralnet": {
        "type_1": 1,
        "type_2": 2
    },
    "type_1": {
        "model_name": {
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
    }
}

class TestFindMatchingModel(unittest.TestCase):

    def test_positive_match(self):
        file_attributes = {
            "neuralnet": "type_1",
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
        self.assertEqual(find_matching_model(file_attributes, mock_data), (True, "model_name"))

    def test_negative_match(self):
        file_attributes = {
            "neuralnet": "type_1",
            "layers": "nonexistent_layer",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
        self.assertEqual(find_matching_model(file_attributes, mock_data), (False, None))

    def test_missing_neuralnet(self):
        file_attributes = {
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
        self.assertEqual(find_matching_model(file_attributes, mock_data), (False, None))

    def test_missing_type_1(self):
        file_attributes = {
            "neuralnet": "type_1",
            "type_1": "nonexistent_type",
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            "hash": "0b0c3b4a0"
        }
        self.assertEqual(find_matching_model(file_attributes, mock_data), (False, None))

    def test_partial_match(self):
        file_attributes = {
            "neuralnet": "type_1",
            "layers": "model.diffusion_model.attn.norm1.weight",
            "tensors": 3,
            "shape": {768, 8},
            # Missing hash attribute
        }
        self.assertEqual(find_matching_model(file_attributes, mock_data), (False, None))

    def test_empty_file_attributes(self):
        file_attributes = {}
        self.assertEqual(find_matching_model(file_attributes, mock_data), (False, None))

if __name__ == "__main__":
    unittest.main()