### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

import unittest
import os

from modules.nnll_48.src import MetadataFileReader


class TestDiskInterface(unittest.TestCase):
    def setUp(self):
        self.reader = MetadataFileReader()
        self.test_folder = os.path.dirname(os.path.abspath(__file__))
        self.real_file = os.path.join(self.test_folder, "test_img.png")

    def test_read_png_header_fail(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.read_header(os.path.join(self.test_folder, "nonexistent.png"))

    def test_read_png_header_succeed(self):
        """Do eeet"""
        chunks = self.reader.read_header(self.real_file)
        self.assertIsNotNone(chunks)
        self.assertTrue(list(chunks))  # Confirm it's not empty


if __name__ == "__main__":
    unittest.main()
