#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s


import unittest
from unittest.mock import patch, MagicMock, mock_open
from collections import defaultdict
import struct
import shutil
import os
import sys

from modules.nnll_05.src import load_gguf_metadata_from_model, read_gguf_header


class TestLoadGGUFMetadata(unittest.TestCase):

    @patch('modules.nnll_05.src.parse_gguf_model')
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
        result = read_gguf_header(self.test_file_name)
        self.assertTrue(result)


    def test_with_file(self):
        try:
            os.environ['HUGGINGFACE_HUB_CACHE'] = str(os.getcwd())
            from huggingface_hub import hf_hub_download
            hf_hub_download("exdysa/tiny-random-llama-gguf","tiny-random-llama.Q4_K_M.gguf")
        except ImportError as error_log:
            ImportError(f"{error_log} huggingface_hub not installed.")
        else:
            self.__class__.folder = os.path.join(str(os.getcwd()), "models--exdysa--tiny-random-llama-gguf")
            real_file = os.path.join(self.__class__.folder,
                "blobs",
                "f06746ef9696d552d3746516558d5e9f338e581fd969158a90824e24f244169c"
                )
        virtual_data_00 = load_gguf_metadata_from_model(real_file)
        print(virtual_data_00)
        self.assertEqual(virtual_data_00,{'name': 'tiny-random-llama', 'dtype': 'float32'})

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove('test.gguf')
        except OSError:
            pass
        try:
            shutil.rmtree(cls.folder)
            shutil.rmtree(".locks")
        except OSError:
            pass