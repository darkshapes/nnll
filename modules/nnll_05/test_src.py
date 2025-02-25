### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest.mock import patch, MagicMock, mock_open
import struct
import shutil
import os

from modules.nnll_05.src import metadata_from_gguf, gguf_check
from modules.nnll_30.src import read_json_file
from modules.nnll_45.src import download_hub_file


class TestLoadMetadataGGUF(unittest.TestCase):
    @patch("modules.nnll_05.src.create_llama_parser")
    def setUp(self, MockParseModel) -> None:
        # Create a temporary file with known GGUF header data
        self.test_file_name = "test.gguf"
        magic = b"GGUF"
        with open(self.test_file_name, "wb") as f:
            f.write(magic)
            f.write(struct.pack("<I", 2))

        # Set up the mock parser object
        self.mock_parser = MagicMock()
        self.mock_parser.metadata = {"general": {"architecture": "Llama", "name": "MyModel"}}
        self.mock_parser.scores = MagicMock(dtype=MagicMock(name="float32"))

        # Make parse_gguf_model return the mock parser
        MockParseModel.return_value = self.mock_parser

    def test_read_valid_header(self):
        result = gguf_check(self.test_file_name)
        self.assertTrue(result)

    def test_metadata_from_gguf(self):
        local_folder = os.path.dirname(os.path.abspath(__file__))
        local_folder_test = os.path.join(local_folder, "test_folder")
        file_name = "TinyStories-LLaMA2-20M-256h-4l-colab.Q2_K.gguf"
        folder_path_named, folder_contents = download_hub_file(repo_id="exdysa/tinystories-llama-20m-gguf", filename=file_name, local_dir=local_folder_test)
        real_file = os.path.join(folder_path_named, file_name)
        virtual_data_00 = metadata_from_gguf(real_file)
        gguf_state_dict = os.path.join(local_folder, "expected_output.json")
        expected_output_part_1 = {"architecture_name": "llama", "general_name": ("TinyStories LLaMA2 20M 256h 4l Colab",)}
        expected_output_part_2 = read_json_file(gguf_state_dict)
        assert virtual_data_00 == (expected_output_part_1, expected_output_part_2)
        try:
            shutil.rmtree(local_folder_test)
            shutil.rmtree(os.path.join(local_folder, ".cache"))
        except OSError:
            pass

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up the temporary file after all tests are done
        try:
            os.remove("test.gguf")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
