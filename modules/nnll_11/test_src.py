
import unittest
from unittest.mock import patch, MagicMock
from transformers import CLIPTokenizer, AutoTokenizer
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler

# Import your method_crafter function from the module where it is defined
from src import method_crafter  # Replace `your_module_name` with the actual name of your module

class TestMethodCrafter(unittest.TestCase):
    @patch('transformers.CLIPTokenizer.from_pretrained')
    def test_valid_tokenizer(self, mock_from_pretrained):
        mock_instance = MagicMock()
        mock_from_pretrained.return_value = mock_instance
        key_class = {"CLIPTOKENIZER": CLIPTokenizer}
        method_name = "from_pretrained"
        location = "exdysa/metadata/tree/main/sdxl-base/tokenizer"
        expressions = {"some_arg": 123}

        result = method_crafter(key_class, method_name, location, expressions)

        mock_from_pretrained.assert_called_once_with(location, some_arg=123)
        self.assertEqual(result, {"CLIPTOKENIZER": mock_instance})

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_valid_tokenizer_with_pretrained(self, mock_from_pretrained):
        mock_instance = MagicMock()
        mock_from_pretrained.return_value = mock_instance
        key_class = {"AUTOTOKENIZER": AutoTokenizer}
        method_name = "from_pretrained"
        location = "exdysa/metadata/tree/main/auraflow-02/tokenizer"
        expressions = {}

        result = method_crafter(key_class, method_name, location, expressions)

        mock_from_pretrained.assert_called_once_with(location)
        self.assertEqual(result, {"AUTOTOKENIZER": mock_instance})

    @patch('diffusers.AutoPipelineForText2Image.from_pretrained')
    def test_valid_pipeline(self, mock_from_pretrained):
        mock_instance = MagicMock()
        mock_from_pretrained.return_value = mock_instance
        key_class = {"AUTOPIPE": AutoPipelineForText2Image}
        method_name = "from_pretrained"
        location = "exdysa/metadata/tree/main/sdxl-base/"
        expressions = {}

        result = method_crafter(key_class, method_name, location, expressions)

        mock_from_pretrained.assert_called_once_with(location)
        self.assertEqual(result, {"AUTOPIPE": mock_instance})

    @patch('diffusers.EulerDiscreteScheduler.from_config')
    def test_valid_scheduler(self, mock_from_config):
        mock_instance = MagicMock()
        mock_from_config.return_value = mock_instance
        key_class = {"EULERDISCRETE": EulerDiscreteScheduler}
        method_name = "from_config"
        location = "exdysa/metadata/tree/main/sdxl-base/"
        expressions = {}

        result = method_crafter(key_class, method_name, location, expressions)

        mock_from_config.assert_called_once_with(location)
        self.assertEqual(result, {"EULERDISCRETE": mock_instance})

    def test_invalid_method(self):
        key_class = {"CLIPTOKENIZER": CLIPTokenizer}
        method_name = "invalid_method"
        location = "/Users/unauthorized/Downloads/models/metadata/CLI-VG"
        expressions = {}

        with self.assertRaises(AttributeError):
            method_crafter(key_class, method_name, location, expressions)

    def test_empty_key_class(self):
        key_class = {}
        method_name = "from_pretrained"
        location = "/Users/unauthorized/Downloads/models/metadata/CLI-VG"
        expressions = {}

        result = method_crafter(key_class, method_name, location, expressions)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()