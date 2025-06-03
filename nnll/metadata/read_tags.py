### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""Wrap Image/Model Metadata I/O"""

# pylint: disable=import-outside-toplevel

from typing import Optional
from nnll.monitor.file import debug_monitor


class MetadataFileReader:
    """Interface for metadata and text read operations"""

    def __init__(self) -> None:
        self.show_content = None
        import nnll.monitor.file as file

        self.nfo = file.nfo
        self.dbug = file.dbug

    @debug_monitor
    def _read_jpg_header(self, file_path_named: str) -> Optional[dict]:
        """
        Open jpg format files\n
        :param file_path_named: The path and file name of the jpg file
        :return: Generator element containing header tags
        """
        from PIL import Image, ExifTags

        img = Image.open(file_path_named)  # pylint: disable=protected-access, line-too-long
        exif_tags = {ExifTags.TAGS[key]: val for key, val in img._getexif().items() if key in ExifTags.TAGS}  # pylint: disable=protected-access, line-too-long
        return exif_tags

    @debug_monitor
    def _read_png_header(self, file_path_named: str) -> Optional[dict]:
        """
        Open png format files\n
        :param file_path_named: The path and file name of the png file
        :return: Generator element containing header tags
        """

        from PIL import Image, UnidentifiedImageError

        try:
            img = Image.open(file_path_named)
            if img is None:  # We dont need to load completely unless totally necessary
                img.load()  # This is the case when we have no choice but to load (slower)
            metadata = img.info
            return metadata  # PNG info directly used here
        except UnidentifiedImageError as error_log:
            self.nfo("Failed to read image at:", file_path_named, error_log)
            return None

    def _read_txt_contents(self, file_path_named: str) -> Optional[dict]:
        """
        Open plaintext files\n
        :param file_path_named: The path and file name of the text file
        :return: Generator element containing content
        """

        try:
            with open(file_path_named, "r", encoding="utf_8") as open_file:
                return open_file.read()  # Reads text file into string
        except UnicodeDecodeError as error_log:
            self.nfo("File did not match expected unicode format %s", file_path_named)
            self.dbug(error_log)
            return None
        try:
            with open(file_path_named, "r", encoding="utf_16-be") as open_file:
                return open_file.read()  # Reads text file into string
        except UnicodeDecodeError as error_log:
            self.nfo("File did not match expected unicode format %s", file_path_named)
            self.dbug(error_log)
            return None

    def _read_schema_file(self, file_path_named: str, mode="r") -> Optional[dict]:
        """
        Open .json or toml files\n
        :param file_path_named: The path and file name of the json file
        :return: Generator element containing content
        """
        import os
        import json
        import tomllib

        from nnll.metadata.constants import ExtensionType as Ext

        _, ext = os.path.splitext(file_path_named)
        if ext in Ext.TOML:
            loader, mode = (tomllib.load, {"mode": "rb"})
        else:
            loader, mode = (json.load, {"model": "r", "encoding": "utf_8"})
        with open(file_path_named, **mode) as open_file:  # pylint:disable=unspecified-encoding
            try:
                return loader(open_file)
            except (tomllib.TOMLDecodeError, json.decoder.JSONDecodeError) as error_log:
                raise SyntaxError(f"Couldn't read file {file_path_named}") from error_log

    @debug_monitor
    def read_header(self, file_path_named: str) -> Optional[dict]:
        """
        Direct file read operations for various file formats\n
        :param file_path_named: Location of file with file name and path
        :return: A mapping of information contained within it
        """
        from nnll.metadata.model_tags import ReadModelTags
        from pathlib import Path
        from nnll.metadata.constants import ExtensionType as Ext

        ext = Path(file_path_named).suffix.lower()
        if ext in Ext.JPEG:
            return self._read_jpg_header(file_path_named)
        if ext in Ext.PNG_:
            return self._read_png_header(file_path_named)
        for file_types in Ext.SCHEMA:
            if ext in file_types:
                return self._read_schema_file(file_path_named)
        for file_types in Ext.PLAIN:
            if ext in file_types:
                return self._read_txt_contents(file_path_named)
        for file_types in Ext.MODEL:
            if ext in file_types:
                model_tool = ReadModelTags()
                return model_tool.read_metadata_from(file_path_named)
