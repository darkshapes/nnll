
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest
from unittest.mock import patch, MagicMock
import os

import os
import sys
from modules.nnll_22.src import AbstractLink, AutoencoderLink, TextEncoderLink, UNetLink


class ConcreteLink(AbstractLink):
    """
    A concrete subclass of AbstractLink for testing purposes.
    """

    def get_filename(self) -> str:
        return "test_model.pth"

    def get_folder_name(self, index: int) -> str:
        return f"folder_{index}"


class TestAbstractLink(unittest.TestCase):

    def test_get_filename_abstract_method(self):
        with self.assertRaises(TypeError):
            abstract_link_instance = AbstractLink()

    def test_get_folder_name_abstract_method(self):
        with self.assertRaises(TypeError):
            abstract_link_instance = AbstractLink()

    @patch('os.path.normpath')
    def test_initialization(self, mock_normpath):
        mock_normpath.return_value = "/Users/unauthorized/Downloads/models/metadata"
        link = ConcreteLink("/Users/unauthorized/Downloads/models/metadata")
        self.assertEqual(link.metadata_folder, "/Users/unauthorized/Downloads/models/metadata")


class TestAutoencoderLink(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_single_file(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = AutoencoderLink()

        # Setup mocks
        mock_islink.return_value = True  # Simulate an existing symlink to be removed
        mock_listdir.return_value = ["old_link"]  # List of files in the directory

        target_path = "/path/to/single_file.bin"
        result = link.create_symlink("model_type", target_path)

        expected_filename = "diffusion_pytorch_model.safetensors"
        expected_folder_name = os.path.join(link.metadata_folder, "model_type")
        expected_symlink_full_path = os.path.join(expected_folder_name, expected_filename)

        # Assertions
        mock_makedirs.assert_called_with(expected_folder_name, exist_ok=True)
        mock_remove.assert_called_with(os.path.join(expected_folder_name, "old_link"))
        mock_symlink.assert_called_with(target_path, expected_symlink_full_path)
        self.assertEqual(result, expected_symlink_full_path)

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_sharded_files(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = AutoencoderLink()

        # Setup mocks
        mock_islink.return_value = True
        mock_listdir.side_effect = [["old_link"], ["old_shard_01"]] * 2

        sharded_files = ["/path/to/shard_01.bin", "/path/to/shard_02.bin"]
        result = link.create_symlink("model_type", sharded_files, original_layout=True)

        expected_filenames = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"]
        expected_folder_name = os.path.join(link.metadata_folder, "model_type", "vae")
        expected_symlink_full_paths = [os.path.join(expected_folder_name, filename) for filename in expected_filenames]

        # Assertions
        mock_makedirs.assert_called_with(os.path.dirname(expected_symlink_full_paths[0]), exist_ok=True)
        self.assertEqual(mock_remove.call_count, 2)
        self.assertEqual(mock_symlink.call_count, 2)

        for i, path in enumerate(sharded_files):
            mock_symlink.assert_any_call(path, expected_symlink_full_paths[i])

        self.assertEqual(result, expected_symlink_full_paths)


class TestTextEncoderLink(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_single_file(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = TextEncoderLink()

        # Setup mocks
        mock_islink.return_value = True
        mock_listdir.return_value = ["old_link"]

        target_path = "/path/to/single_file.bin"
        result = link.create_symlink("model_type", target_path)

        expected_filename = "model.safetensors"
        expected_folder_name = os.path.join(link.metadata_folder, "model_type")
        expected_symlink_full_path = os.path.join(expected_folder_name, expected_filename)

        # Assertions
        mock_makedirs.assert_called_with(expected_folder_name, exist_ok=True)
        mock_remove.assert_called_with(os.path.join(expected_folder_name, "old_link"))
        mock_symlink.assert_called_with(target_path, expected_symlink_full_path)
        self.assertEqual(result, expected_symlink_full_path)

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_sharded_files(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = TextEncoderLink()

        # Setup mocks
        mock_islink.return_value = True
        mock_listdir.side_effect = [["old_link"], ["old_shard_01"]] * 2

        sharded_files = ["/path/to/shard_01.bin", "/path/to/shard_02.bin"]
        result = link.create_symlink("model_type", sharded_files, original_layout=True)

        expected_filenames = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        expected_folder_name = os.path.join(link.metadata_folder, "model_type", "text_encoder")
        expected_symlink_full_paths = [os.path.join(expected_folder_name, filename) for filename in expected_filenames]

        # Assertions
        mock_makedirs.assert_called_with(os.path.dirname(expected_symlink_full_paths[0]), exist_ok=True)
        self.assertEqual(mock_remove.call_count, 2)
        self.assertEqual(mock_symlink.call_count, 2)

        for i, path in enumerate(sharded_files):
            mock_symlink.assert_any_call(path, expected_symlink_full_paths[i])

        self.assertEqual(result, expected_symlink_full_paths)


class TestUNetLink(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_single_file(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = UNetLink()

        # Setup mocks
        mock_islink.return_value = True
        mock_listdir.return_value = ["old_link"]

        target_path = "/path/to/single_file.bin"
        result = link.create_symlink("model_type", target_path)

        expected_filename = "diffusion_pytorch_model.safetensors"
        expected_folder_name = os.path.join(link.metadata_folder, "model_type")
        expected_symlink_full_path = os.path.join(expected_folder_name, expected_filename)

        # Assertions
        mock_makedirs.assert_called_with(expected_folder_name, exist_ok=True)
        mock_remove.assert_called_with(os.path.join(expected_folder_name, "old_link"))
        mock_symlink.assert_called_with(target_path, expected_symlink_full_path)
        self.assertEqual(result, expected_symlink_full_path)

    @patch('os.makedirs')
    @patch('os.listdir')
    @patch('os.remove')
    @patch('os.symlink')
    @patch('os.path.islink')
    def test_create_symlink_sharded_files(self, mock_islink, mock_symlink, mock_remove, mock_listdir, mock_makedirs):
        link = UNetLink()

        # Setup mocks
        mock_islink.return_value = True
        mock_listdir.side_effect = [["old_link"], ["old_shard_01"]] * 2

        sharded_files = ["/path/to/shard_01.bin", "/path/to/shard_02.bin"]
        result = link.create_symlink("model_type", sharded_files, original_layout=True)

        expected_filenames = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"]
        expected_folder_name = os.path.join(link.metadata_folder, "model_type", "unet")
        expected_symlink_full_paths = [os.path.join(expected_folder_name, filename) for filename in expected_filenames]

        # Assertions
        mock_makedirs.assert_called_with(os.path.dirname(expected_symlink_full_paths[0]), exist_ok=True)
        self.assertEqual(mock_remove.call_count, 2)
        self.assertEqual(mock_symlink.call_count, 2)

        for i, path in enumerate(sharded_files):
            mock_symlink.assert_any_call(path, expected_symlink_full_paths[i])

        self.assertEqual(result, expected_symlink_full_paths)

if __name__== "__main__":
    text_link = TextEncoderLink()
    unet_link = UNetLink()
    autoencoder_link = AutoencoderLink()

    target_path_text = sys.argv[0]
    shards_text = ["/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.1of2safetensors.safetensors", "/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.2of2safetensors.safetensors"]

    print(text_link.create_symlink(model_type="clip-l", target_path=target_path_text))
    print(text_link.create_symlink(model_type="t5-xxl", target_path=shards_text))
