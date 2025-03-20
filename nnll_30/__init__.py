### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def write_json_file(file_path: str, file_name: str, data, mode: str = "w"):
    import os
    import json

    if ".json" not in file_name:
        file_name += ".json"
    document = os.path.join(file_path, os.path.basename(file_name))
    with open(document, mode, encoding="UTF-8") as i:
        json.dump(data, i, ensure_ascii=False, indent=4, sort_keys=False)


def read_json_file(file_path: str, mode="r"):
    import json

    with open(file_path, mode, encoding="UTF-8") as f:
        return json.load(f)
