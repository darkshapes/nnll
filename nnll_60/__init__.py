### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""JSONCache, HASH_PATH_NAMED,CONFIG_PATH_NAMED,CHAIN_PATH_NAMED"""

import os

HASH_PATH_NAMED = os.path.join(os.path.dirname(__file__), "config", "hashes.json")
CONFIG_PATH_NAMED = os.path.join(os.path.dirname(__file__), "config", "config.json")
CHAIN_PATH_NAMED = os.path.join(os.path.dirname(__file__), "config", "hyperchain.json")
LIBTYPE_PATH_NAMED = os.path.join(os.path.dirname(__file__), "config", "libtype.json")


class JSONCache:
    """Manage input/output disk/mem for json files"""

    def __init__(self, file_or_path: str):
        """Cache operations for .json files. Example:
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
        """Populate cache with file data if not already populated"""
        import json

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
        import json

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
