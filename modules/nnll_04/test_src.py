
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s
import unittest
import os
import shutil

from modules.nnll_04.src import metadata_from_safetensors
from modules.nnll_30.src import read_json_file
from modules.nnll_45.src import download_hub_file


class TestLoadMetadataSafetensors(unittest.TestCase):

    def test_metadata_from_safetensors(self):
        self.folder_path_named, folder_contents = download_hub_file(repo_id='exdysa/tiny-random-gpt2-safetensors',filename='model.safetensors')
        real_file = os.path.join(self.folder_path_named, 'blobs', next(iter(folder_contents)))
        virtual_data_00 = metadata_from_safetensors(real_file)
        expected_output = read_json_file(os.path.join(os.getcwd(),'modules','nnll_04','expected_output.json'))
        assert virtual_data_00 == expected_output
        try:
            shutil.rmtree(self.folder_path_named)
            shutil.rmtree(".locks")
        except OSError:
            pass

if __name__ == '__main__':
    unittest.main()
