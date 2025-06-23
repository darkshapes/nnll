### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

import unittest
import os
import shutil

from nnll.metadata.model_tags import ReadModelTags
from nnll.metadata.json_io import read_json_file
from nnll.download.hub_cache import download_hub_file


class TestLoadMetadataSafetensors(unittest.TestCase):
    def test_metadata_from_safetensors(self):
        model_tool = ReadModelTags()
        local_folder = os.path.dirname(os.path.abspath(__file__))
        local_folder_test = os.path.join(local_folder, "test_folder")
        file_name = "model.safetensors"
        folder_path_named, _ = download_hub_file(repo_id="exdysa/tiny-random-gpt2-bfloat16", filename=file_name, local_dir=local_folder_test)
        real_file = os.path.join(folder_path_named, file_name)
        virtual_data_00 = model_tool.metadata_from_safetensors(real_file)
        safetensors_state_dict = os.path.join(local_folder, "test_safetensor_tag_expected.json")
        expected_output = read_json_file(safetensors_state_dict)
        assert virtual_data_00 == expected_output
        try:
            shutil.rmtree(local_folder_test)
            shutil.rmtree(os.path.join(local_folder, ".cache"))
        except OSError:
            pass


if __name__ == "__main__":
    # unittest.main()
    import pytest

    pytest.main(["-vv", __file__])
