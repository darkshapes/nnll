# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

"""為使用者介面清理和安排元資料"""

# pylint: disable=line-too-long

import re
import json
from json import JSONDecodeError
from typing import Tuple, List
from pydantic import ValidationError

from dataset_tools.logger import debug_monitor  # , debug_message
from dataset_tools.logger import info_monitor as nfo
from dataset_tools.access_disk import MetadataFileReader
from dataset_tools.correct_types import (
    IsThisNode,
    NodeWorkflow,
    BracketedDict,
    ListOfDelineatedStr,
    UpField,
    DownField,
    NodeNames,
)


# /______________________________________________________________________________________________________________________ Module Interface


@debug_monitor
def coordinate_metadata_ops(header_data: dict | str, metadata: dict = None) -> dict:
    """
    Process data based on identifying contents\n
    :param header_data: Metadata extracted from file
    :type header_data: dict
    :param datatype: The kind of variable storing the metadata
    :type datatype: str
    :param metadata: The filtered output extracted from header data
    :type metadata: dict
    :return: A dict of the metadata inside header data
    """

    has_prompt = isinstance(header_data, dict) and header_data.get("prompt")
    has_params = isinstance(header_data, dict) and header_data.get("parameters")
    has_tags = isinstance(header_data, dict) and ("icc_profile" in header_data or "exif" in header_data)

    if has_prompt:
        metadata = arrange_nodeui_metadata(header_data)
    elif has_params:
        metadata = arrange_webui_metadata(header_data)
    elif has_tags:
        metadata = arrange_exif_metadata(header_data)
    elif isinstance(header_data, dict):
        try:
            metadata = {UpField.JSON_DATA: json.loads(f"{header_data}")}
        except JSONDecodeError as error_log:
            nfo("JSON Decode failed %s", error_log)
    if not metadata and isinstance(header_data, str):
        metadata = {UpField.DATA: header_data}
    elif not metadata:
        metadata = {UpField.PLACEHOLDER: {"": UpField.PLACEHOLDER}}

    return metadata


@debug_monitor
def parse_metadata(file_path_named: str) -> dict:
    """
    Extract the metadata from the header of an image file\n
    :param file_path_named: The file to open
    :return: The metadata from the header of the file
    """

    reader = MetadataFileReader()
    metadata = None
    header_data = reader.read_header(file_path_named)
    if header_data is not None:
        metadata = coordinate_metadata_ops(header_data)
        if metadata == {UpField.PLACEHOLDER: {"": UpField.PLACEHOLDER}} or not isinstance(metadata, dict):
            nfo("Unexpected format", file_path_named)
    return metadata
