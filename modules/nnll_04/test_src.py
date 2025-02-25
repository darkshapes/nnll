### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

import unittest
import os
import shutil

from modules.nnll_04.src import metadata_from_safetensors
from modules.nnll_30.src import read_json_file
from modules.nnll_45.src import download_hub_file


class TestLoadMetadataSafetensors(unittest.TestCase):
    def test_metadata_from_safetensors(self):
        local_folder = os.path.dirname(os.path.abspath(__file__))
        local_folder_test = os.path.join(local_folder, "test_folder")
        file_name = "model.safetensors"
        folder_path_named, folder_contents = download_hub_file(repo_id="exdysa/tiny-random-gpt2-bfloat16", filename=file_name, local_dir=local_folder_test)
        real_file = os.path.join(folder_path_named, file_name)
        virtual_data_00 = metadata_from_safetensors(real_file)
        safetensors_state_dict = os.path.join(local_folder, "expected_output.json")
        expected_output = read_json_file(safetensors_state_dict)
        assert virtual_data_00 == expected_output
        try:
            shutil.rmtree(local_folder_test)
            shutil.rmtree(os.path.join(local_folder, ".cache"))
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
