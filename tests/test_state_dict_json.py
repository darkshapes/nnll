# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import shutil
import unittest
import os
from nnll.download.hub_cache import download_hub_file
from nnll.configure import ensure_path
from nnll.metadata.json_io import write_json_file


class TestReadStateDictHeaders(unittest.TestCase):
    local_test_folder = os.path.dirname(os.path.abspath(__file__))
    temp_folder = str(ensure_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_folder")))

    def setUp(self):
        self.temp_folder = self.temp_folder
        self.local_test_folder = self.local_test_folder

    def test_read_state_dict_headers(self):
        # Mock Path objects for each file with correct suffixes
        safetensors_folder_path_named, safetensors_file = download_hub_file(repo_id="exdysa/RA-SAE-DINOv2-32k", filename="model.safetensors", local_dir=self.temp_folder)
        gguf_folder__path_named, gguf_file = download_hub_file(repo_id="exdysa/ratchet-test", filename="dummy.gguf", local_dir=self.temp_folder)

        # Call the function under test
        from nnll.metadata.model_tags import ReadModelTags

        reader = ReadModelTags()
        for file in os.listdir(gguf_folder__path_named):
            metadata = reader.read_metadata_from(os.path.join(gguf_folder__path_named, file))
            write_json_file(self.temp_folder, f"{os.path.dirname(gguf_folder__path_named)}_{os.path.basename(gguf_folder__path_named)}_{file}.json", metadata)

    def tearDown(cls):
        try:
            shutil.rmtree(cls.temp_folder)
            shutil.rmtree(".locks")
        except OSError:
            pass
        try:
            shutil.rmtree(cls.temp_folder)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
