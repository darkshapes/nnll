# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""JSONCache, [   ] _PATH_NAMED"""

import os
from logging import INFO, Logger
from pathlib import Path
from typing import Union

from nnll.integrity import ensure_path

nfo_obj = Logger(INFO)
nfo = nfo_obj.info


def set_path_stable(file_name: str, folder_path: str = os.path.dirname(__file__), prefix: str = "config") -> Path:
    """Create a constant for a given path, or the path of the current file\n
    :param file_name: The tail/basename of the absolute file path to
    :param folder_path: The head of the absolute file path, defaults to `os.path.dirname(__file__)`
    :param prefix: Optional folder between `folder_path` and `file_name`, defaults to "config"
    :return: A combined path string of the given values
    """
    folder_path_named = os.path.join(folder_path, prefix, file_name)
    Path(folder_path_named).touch()
    return folder_path_named
    # return ensure_path(folder_path_named, file_name)


constants = [
    file_name.stem
    for file_name in Path(os.path.join(os.path.dirname(__file__), "config")).iterdir()
    if "__" not in Path(file_name).stem
    # comment for formatting
]

for const in constants:
    paths = {}
    path_var = f"{const.upper()}_PATH_NAMED"
    globals()[path_var] = set_path_stable(const + ".json")


class JSONCache:
    """Manage input/output disk/mem for json and read-only toml files"""

    def __init__(self, file_path_named: Union[str, Path]):
        """Cache operations for .json and read-only .toml files. Example:
        ```
        cache_manager = JSONCache("path/to/file.json")

        @cache_manager.decorator
        #def docstring_demo_func(data):
            print(data)
        ```
        Force save:
        `cache_manager.update({"new_key": "new_value"})`
        """

        self.file: Union[str, Path] = file_path_named
        self._cache: dict = {}

    def _load_cache(self):
        """Populate cache with **text** file data if not already populated"""
        import json
        import tomllib

        # dbuq(f"loading_file {self.file}")

        if not self._cache:
            if Path(self.file).suffix.lower() == ".toml":
                with open(self.file, "rb") as f:
                    try:
                        self._cache = tomllib.load(f)
                    except tomllib.TOMLDecodeError:  ## as error_log:
                        nfo("Error decoding cache file", f" {self.file} Using an empty cache.")
                        # dbuq(f"Error decoding cache file. Using an empty cache. {error_log}")
                        self._cache = {}
            else:
                try:
                    with open(self.file, "r", encoding="UTF-8") as f:
                        self._cache = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):  # as error_log:
                    nfo("Error decoding cache file", f" {self.file} Using an empty cache.")
                    # dbuq(f"Error decoding cache file. Using an empty cache. {error_log}")
                    self._cache = {}

    def _save_cache(self):
        """
        Write data to a temporary file to minimize data corruption.
        Replace the original file with the updated cache.
        """
        import json

        temp_file = str(self.file) + ".tmp"
        with open(temp_file, "w", encoding="UTF-8") as doc:
            if Path(self.file).suffix == ".toml":
                pass
            else:
                json.dump(self._cache, doc, ensure_ascii=False, indent=4)
        os.replace(temp_file, self.file)

    def refresh(self):
        """External trigger for manual cache retrieval"""
        self._load_cache()

    def update_cache(self, new_data: dict, replace: bool = False):
        """Save changes only if data actually changed
        :param new_data: Updated dictionary
        :param replace: Force clear entire cache and replace with new_data, defaults to False
        """
        if replace:
            self._cache = {"empty": ""}  # sanity check
            self._save_cache()
            self._cache.pop("empty")
            os.remove(self.file)
            Path(self.file).touch()
        self._load_cache()  # Ensure cache loaded / 確保快取載入
        original_cache_copy = self._cache.copy()  # Snapshot current state / 快照當前快取
        self._cache.update(new_data)  # Add the data to the cache / 將資料新增到快取中
        if original_cache_copy != self._cache:
            self._save_cache()

    def decorator(self, func):
        """Add cache file copies to functions"""
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Triggers cache read when called, feeds it to calling function.
            Not yet implemented: trigger save automatically with data discrepancy
            """
            self._load_cache()

            updated_data = kwargs.pop("data", {})
            original_cache_copy = self._cache.copy()

            if updated_data:
                self._cache.update(updated_data)

                # Save only if cache has changed
                if original_cache_copy != self._cache:
                    self._save_cache()

            return func(*args, **kwargs, data=self._cache)

        return wrapper
