### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
import os
import shutil
from nnll_36 import read_state_dict_headers
from nnll_45 import download_hub_file


class TestReadStateDictHeaders(unittest.TestCase):
    def test_read_state_dict_headers(self):
        # Mock Path objects for each file with correct suffixes
        safetensors_folder_path_named, safetensors_file = download_hub_file(repo_id="exdysa/tiny-random-gpt2-bfloat16", filename="model.safetensors", local_dir="./test_folder")
        gguf_folder__path_named, gguf_file = download_hub_file(repo_id="exdysa/tinystories-llama-20m-gguf", filename="TinyStories-LLaMA2-20M-256h-4l-colab.Q2_K.gguf", local_dir="./test_folder")

        # Call the function under test
        read_state_dict_headers("test_folder", "test_folder")

        self.assertIsNone(read_state_dict_headers(folder_path_named="test_folder", save_location="./test_folder"))

        try:
            shutil.rmtree(gguf_folder__path_named)
            shutil.rmtree(".locks")
        except OSError:
            pass
        try:
            shutil.rmtree(safetensors_folder_path_named)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
