### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""JSONCache, HASH_PATH_NAMED,CONFIG_PATH_NAMED,CHAIN_PATH_NAMED USER_PATH_NAMED"""

import os
import sys

sys.path.append(os.getcwd())
from pathlib import Path
from functools import cache
from nnll_01 import dbug


@cache
def set_home_stable() -> Path:
    """Return platform-dependent app data mapping\n
    :return: A platform-specific path to vendor-designated app data folder\n
    RATIONALE: operator may want to discard the application\n
    EXAMPLES: to maintain experimental conditions, improper venv setup, conflicting dependencies, troubleshooting,
    overreliance on reinstalling to fix things, switching computersquit, disk space full, they got advice online, etc.\n Therefore,
    To accomodate user so they can return to previous settings, leverage os-specific library location.
    """
    from platform import system

    return (
        os.path.join(os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local")), "Shadowbox")
        if system().lower() == "windows"
        else os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Shadowbox")
        if system().lower() == "darwin"
        else os.path.join(os.path.expanduser("~"), ".config", "shadowbox")
    )


HOME_FOLDER_PATH = set_home_stable()


def set_path_stable(file_name: str, folder_path: str = os.path.dirname(__file__), prefix: str = "config") -> Path:
    """Create a constant for a given path, or the path of the current file\n
    :param file_name: The tail/basename of the absolute file path to
    :param folder_path: The head of the absolute file path, defaults to `os.path.dirname(__file__)`
    :param prefix: Optional folder between `folder_path` and `file_name`, defaults to "config"
    :return: A combined path string of the given values
    """

    return os.path.join(folder_path, prefix, file_name)


USER_PATH_NAMED = os.path.join(HOME_FOLDER_PATH, "config.toml")
HASH_PATH_NAMED = set_path_stable("hashes.json")
# CONFIG_PATH_NAMED = set_path_stable("config.json")
CHAIN_PATH_NAMED = set_path_stable("hyperchain.json")
LIBTYPE_PATH_NAMED = set_path_stable("libtype.json")
MIR_PATH = set_path_stable("mir.json")


class JSONCache:
    """Manage input/output disk/mem for json and read-only toml files"""

    def __init__(self, file_or_path: str):
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

        self.file: str = file_or_path
        self._cache: dict = {}

    def _load_cache(self):
        """Populate cache with **text** file data if not already populated"""
        import json
        import tomllib

        if not self._cache:
            if Path(self.file).suffix.lower() == ".toml":
                with open(self.file, "rb") as f:
                    try:
                        self._cache = tomllib.load(f)
                    except tomllib.TOMLDecodeError as error_log:
                        dbug(f"Error decoding cache file. Using an empty cache. {error_log}")
                        self._cache = {}
            else:
                with open(self.file, "r", encoding="UTF-8") as f:
                    try:
                        self._cache = json.load(f)
                    except FileNotFoundError:
                        self._cache = {}
                    except json.JSONDecodeError as error_log:
                        dbug(f"Error decoding cache file. Using an empty cache. {error_log}")
                        self._cache = {}

    def _save_cache(self):
        """
        Write data to a temporary file to minimize data corruption.
        Replace the original file with the updated cache.
        """
        import json

        temp_file = self.file + ".tmp"
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
        self._load_cache()  # Ensure cache loaded / 確保快取載入
        if replace:
            self._cache = {"empty": ""}  # sanity check
            self._save_cache()
            self._cache.pop("empty")
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
