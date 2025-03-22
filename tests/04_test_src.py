### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

import unittest
import os
import shutil

from nnll_04 import ModelTool
from nnll_30 import read_json_file
from nnll_45 import download_hub_file


class TestLoadMetadataSafetensors(unittest.TestCase):
    def test_metadata_from_safetensors(self):
        model_tool = ModelTool()
        local_folder = os.path.dirname(os.path.abspath(__file__))
        local_folder_test = os.path.join(local_folder, "test_folder")
        file_name = "model.safetensors"
        folder_path_named, folder_contents = download_hub_file(repo_id="exdysa/tiny-random-gpt2-bfloat16", filename=file_name, local_dir=local_folder_test)
        real_file = os.path.join(folder_path_named, file_name)
        virtual_data_00 = model_tool.metadata_from_safetensors(real_file)
        safetensors_state_dict = os.path.join(local_folder, "04_expected_output.json")
        expected_output = read_json_file(safetensors_state_dict)
        assert virtual_data_00 == expected_output
        try:
            shutil.rmtree(local_folder_test)
            shutil.rmtree(os.path.join(local_folder, ".cache"))
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
