# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import shutil
import unittest

from nnll.download.hub_cache import download_hub_file
from nnll.metadata.model_tags import main


class TestReadStateDictHeaders(unittest.TestCase):
    def test_read_state_dict_headers(self):
        # Mock Path objects for each file with correct suffixes
        safetensors_folder_path_named, safetensors_file = download_hub_file(repo_id="exdysa/RA-SAE-DINOv2-32k", filename="model.safetensors", local_dir="./test_folder")
        gguf_folder__path_named, gguf_file = download_hub_file(repo_id="exdysa/ratchet-test", filename="dummy.gguf", local_dir="./test_folder")

        # Call the function under test

        main("test_folder", "test_folder")

        self.assertIsNone(main(folder_path_named="test_folder", save_location="./test_folder"))

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
