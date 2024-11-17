
import os
import struct
from collections import defaultdict
from unittest.mock import patch, MagicMock
import unittest
from src import ungguf

# Mocking Llama class and its methods for testing purposes


class MockLlama:
    def __init__(self, model_path, vocab_only, verbose):
        self.model_path = model_path
        self.vocab_only = vocab_only
        self.verbose = verbose
        self.metadata = {}
        self.scores = MagicMock()
        self.scores.dtype = MagicMock(name='dtype')
        self.scores.dtype.name = 'float32'

    def load_metadata(self, metadata):
        self.metadata.update(metadata)

# Mocking os.path.getsize for testing purposes


def mock_getsize(file_name):
    return 1024  # Simulate a file size of 1024 bytes


class TestUnggufFunction(unittest.TestCase):

    @patch('os.path.getsize', side_effect=mock_getsize)
    def test_valid_file(self, mock_getsize):
        # Create a temporary file with valid GGUF magic number and version
        with open("test.gguf", "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 2))  # Version 2

        parser = MockLlama(model_path="test.gguf", vocab_only=True, verbose=False)
        parser.load_metadata({"general.architecture": "LLaMA-7B", "general.name": "LLaMA-7B"})

        with patch('src.Llama', return_value=parser):
            id_values = {}
            result = ungguf("test.gguf", id_values)

        self.assertEqual(result["file_size"], 1024)
        self.assertEqual(result["name"], "LLaMA-7B")
        self.assertEqual(result["dtype"], "float32")

    @patch('os.path.getsize', side_effect=mock_getsize)
    def test_invalid_magic_number(self, mock_getsize):
        # Create a temporary file with invalid GGUF magic number
        with open("test.gguf", "wb") as f:
            f.write(b"GGXX")
            f.write(struct.pack("<I", 2))  # Version 2

        id_values = {}
        result = ungguf("test.gguf", id_values)

        self.assertIsNone(result)

    @patch('os.path.getsize', side_effect=mock_getsize)
    def test_unsupported_version(self, mock_getsize):
        # Create a temporary file with unsupported GGUF version
        with open("test.gguf", "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))  # Version 1

        id_values = {}
        result = ungguf("test.gguf", id_values)

        self.assertIsNone(result)

    @patch('os.path.getsize', side_effect=mock_getsize)
    def test_exception_handling(self, mock_getsize):
        # Create a temporary file that will raise an exception when opened
        with open("test.gguf", "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 2))  # Version 2

        id_values = {}
        with patch('src.Llama', side_effect=ValueError("Mocked ValueError")):
            result = ungguf("test.gguf", id_values)

        self.assertIsNone(result)

    def tearDown(self):
        # Clean up temporary files after each test
        if os.path.exists("test.gguf"):
            os.remove("test.gguf")


# Run the tests
if __name__ == '__main__':
    unittest.main()
