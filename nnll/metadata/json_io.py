# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Any


def write_json_file(folder_path_named: str, file_name: str, data: Any, mode: str = "tw") -> None:
    """Save a file to disk as UTF8 JSON\n
    :param folder_path_named: The location to save
    :param file_name: A file name to save with
    :param data: The content to save
    :param mode: Type of open to use (default truncate-write)
    :returns: Dictionary of json data
    """
    import json
    import os

    from nnll.monitor.file import dbug

    if ".json" not in file_name:
        file_name += ".json"
    document = os.path.join(folder_path_named, os.path.basename(file_name))
    if os.path.exists(document):
        try:
            os.remove(document)
        except FileNotFoundError as error_log:
            dbug(f"'File was detected but not found to remove: {document}.'{error_log}", exc_info=True)

    with open(document, mode, encoding="UTF-8") as i:
        json.dump(data, i, ensure_ascii=False, indent=4, sort_keys=False)


def read_json_file(file_path_absolute: str, mode="tr") -> dict:
    """Open json file as UTF8 JSON
    :param file_path_absolute: Location of the file
    :param mode: Type of read to use
    :returns: Dictionary of json data
    """
    import json

    with open(file_path_absolute, mode, encoding="UTF-8") as f:
        return json.load(f)
