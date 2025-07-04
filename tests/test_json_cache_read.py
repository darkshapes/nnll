# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint:disable=protected-access, unsubscriptable-object

import unittest
import json
import os
from nnll.mir.json_cache import JSONCache


class TestJSONCache(unittest.TestCase):
    def setUp(self):
        """Set up a temporary JSON file for testing."""
        self.test_file = "test_cache.json"
        self.test_data = {"key": "value"}

        with open(self.test_file, "w", encoding="UTF-8") as f:
            json.dump(self.test_data, f)

        self.cache = JSONCache(self.test_file)

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file + ".tmp"):
            os.remove(self.test_file + ".tmp")

    def test_initialization(self):
        """Ensure JSONCache initializes correctly."""
        self.assertEqual(self.cache.file, self.test_file)
        self.assertEqual(self.cache._cache, {})

    def test_load_cache(self):
        """Check if cache loads properly from a file."""
        self.cache.refresh()
        self.assertEqual(self.cache._cache, self.test_data)

    def test_update_cache(self):
        """Verify cache updates and writes to file."""
        new_data = {"new_key": "new_value"}
        self.cache.update_cache(new_data)
        self.cache.refresh()
        self.assertIn("new_key", self.cache._cache)
        self.assertEqual(self.cache._cache["new_key"], "new_value")

        with open(self.test_file, "r", encoding="UTF-8") as f:
            file_data = json.load(f)
        self.assertEqual(file_data["new_key"], "new_value")

    def test_refresh_cache(self):
        """Ensure manual refresh retrieves latest data."""
        self.cache.refresh()
        self.assertEqual(self.cache._cache, self.test_data)

        # Manually modify the file
        modified_data = {"modified_key": "modified_value"}

        @self.cache.decorator
        def sample_function(data=None):
            return data

        data = sample_function()
        print(data)

        self.cache.refresh()  # ensure the data has not been manipulated yet
        self.assertEqual(self.cache._cache, self.test_data)

        data.update(modified_data)  # ensure the data has been manipulated
        self.assertEqual(self.cache._cache, modified_data | self.test_data)

    def test_decorator_functionality(self):
        """Check that the decorator loads cache and updates properly."""
        self.cache.update_cache({"test_key": "test_value"})

        @self.cache.decorator
        def sample_function(data=None):
            return data

        result = sample_function()
        self.assertIn("test_key", result)
        self.assertEqual(result["test_key"], "test_value")

        # Check update via decorator
        sample_function(data={"another_key": "another_value"})
        self.cache.refresh()
        self.assertIn("another_key", self.cache._cache)
        self.assertEqual(self.cache._cache["another_key"], "another_value")


if __name__ == "__main__":
    unittest.main()
