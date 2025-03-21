### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
from unittest.mock import patch, MagicMock
import nnll_57
from nnll_57 import save_element, write_to_disk
from PIL import Image

mock_path = os.path.dirname(nnll_57.__file__)


def test_save_element():
    with patch("os.path.isdir", return_value=True):
        path = save_element(".png")
        assert path == os.path.join(mock_path, "Combo_000000.png")


def test_write_to_disk():
    image = Image.new("RGB", (60, 30), color="red")
    metadata = {"Test": "Data"}
    with patch("os.listdir") as mock_listdir:
        mock_listdir.return_value = ["file1.png"]
        with patch("PIL.Image.Image.save") as mock_save:
            write_to_disk(image, metadata)
            mock_save.assert_called_with(os.path.join(mock_path, "Combo_000001.png"), "PNG", pnginfo=metadata)
