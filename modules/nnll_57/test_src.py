import os
from unittest.mock import patch, MagicMock
import modules.nnll_57.src as main
from PIL import Image

mock_path = os.path.join(os.getcwd(), "modules", "nnll_57")


def test_save_element():
    with patch("os.path.isdir", return_value=True):
        with patch.object(main.look, "save_folder_name_and_path", new=mock_path):
            path = main.save_element(".png")
            assert path == os.path.join(mock_path, "Combo_000000.png")


def test_write_to_disk():
    image = Image.new("RGB", (60, 30), color="red")
    metadata = {"Test": "Data"}
    with patch("os.listdir") as mock_listdir:
        mock_listdir.return_value = ["file1.png"]
        with patch("PIL.Image.Image.save") as mock_save:
            main.write_to_disk(image, metadata)
            mock_save.assert_called_with(os.path.join(mock_path, "Combo_000001.png"), "PNG", pnginfo=metadata)
