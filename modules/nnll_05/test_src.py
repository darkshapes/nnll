
import unittest
from unittest.mock import patch, MagicMock, mock_open
from collections import defaultdict
import os
import struct

from nnll_05.src import load_gguf_metadata, read_gguf_header


class TestLoadGGUFMetadata(unittest.TestCase):

    @patch('nnll_05.src.parse_gguf_model')
    def setUp(self, MockParseModel) -> None:
        # Create a temporary file with known GGUF header data
        self.test_file_name = 'test.gguf'
        magic = b'GGUF'
        with open(self.test_file_name, 'wb') as f:
            f.write(magic)
            f.write(struct.pack('<I', 2))

        # Set up the mock parser object
        self.mock_parser = MagicMock()
        self.mock_parser.metadata = {
            "general": {
                "architecture": "Llama",
                "name": "MyModel"
            }
        }
        self.mock_parser.scores = MagicMock(dtype=MagicMock(name='float32'))

        # Make parse_gguf_model return the mock parser
        MockParseModel.return_value = self.mock_parser

    def test_read_valid_header(self):
        file_name = 'test.gguf'
        result = read_gguf_header(self.test_file_name)
        self.assertEqual(result, (b'GGUF', 2))

    def test_with_file(self):
        id_values_00 = defaultdict(dict)
        file_name = "/Users/unauthorized/Downloads/models/text/lightblue-ao-karasu-72B-Q4_K_M.gguf"
        virtual_data_00 = load_gguf_metadata(file_name)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove('test.gguf')
        except OSError:
            pass
