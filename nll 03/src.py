import unittest

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

def find_matching_model(file_attributes, data_dict):
    for key, value in data_dict.items():
        if key == file_attributes.get("neuralnet"):
            if "type_1" in file_attributes:
                if file_attributes["type_1"] in data_dict:
                    for model_name, model_data in data_dict[file_attributes["type_1"]].items():
                        if all(attr in model_data.values() for attr in file_attributes.values()):
                            return True, model_name  # Return both True and the matching model name.
    return False, None

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

if __name__ == '__main__':
    unittest.main()