### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import json
from functools import wraps

CONFIG_PATH_NAMED = os.path.join(os.getcwd(), "modules", "nnll_60", "config", "config.json")
CHAIN_PATH_NAMED = os.path.join(os.getcwd(), "modules", "nnll_60", "config", "hyperchain.json")


class JSONCache:
    """Manage input/output disk/mem for json files"""

    def __init__(self, file_or_path: str):
        """Cache operations for .json files. Example:
        ```
        cache_manager = JSONCache("path/to/file.json")

        @cache_manager.decorator`
        #def docstring_demo_func(data):
            print(data)
        ```
        Force save:
        `cache_manager.update({"new_key": "new_value"})`
        """

        self.file = file_or_path
        self._cache = None

    def _load_cache(self):
        """Populate cache with file data if not already populated"""
        if not self._cache:
            with open(self.file, "r", encoding="UTF-8") as f:
                try:
                    self._cache = json.load(f)
                except FileNotFoundError:
                    self._cache = {}
                except json.JSONDecodeError:
                    # print("Error decoding JSON. Using an empty cache.")
                    self._cache = {}

    def _save_cache(self):
        """
        Write data to a temporary file to minimize data corruption.
        Replace the original file with the updated cache.
        """
        temp_file = self.file + ".tmp"
        with open(temp_file, "w", encoding="UTF-8") as doc:
            json.dump(self._cache, doc, ensure_ascii=False, indent=4)
        os.replace(temp_file, self.file)

    def refresh(self):
        """External trigger for manual cache retrieval"""
        self._load_cache()

    def update_cache(self, new_data: dict):
        """Save changes if the data actually changed"""
        self._load_cache()  # Ensure cache loaded / 確保快取載入

        original_cache_copy = self._cache.copy()  # Snapshot current state / 快照當前快取
        self._cache.update(new_data)  # Add the data to the cache/ 將資料新增到快取中

        if original_cache_copy != self._cache:
            self._save_cache()

    def decorator(self, func):
        """Add cache file copies to functions"""

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
