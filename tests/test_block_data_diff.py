# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Tests `block_data_diff` method.
The method is expected to:
1. Follow an existing index pointer (rule1).
2. Create a new index pointer when the incoming value matches the previous block (rule2).
3. Leave the value unchanged when it differs (rule3).
"""

# pylint:disable=line-too-long
import unittest
from unittest.mock import patch
import json
import os

from nnll.hyperchain import HyperChain
from nnll.reverse_codec import ReversibleBytes
from nnll.block import Block


class TestBlockDataDiff(unittest.TestCase):
    """Test block_data_diff method with three rules"""

    test_file = "test_block_data_diff.json"

    @classmethod
    def setUpClass(cls):
        """Set up test file and patch HyperChain"""
        try:
            os.remove(cls.test_file)
        except FileNotFoundError:
            pass
        with open(cls.test_file, "w", encoding="UTF-8") as empty_file:
            json.dump({}, empty_file)
        cls.patcher = patch.object(HyperChain, "chain_file", new=cls.test_file)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        """Clean up patch"""
        cls.patcher.stop()
        try:
            os.remove(cls.test_file)
        except FileNotFoundError:
            pass

    def setUp(self):
        """Create a fresh HyperChain for each test"""
        with open(self.test_file, "w", encoding="UTF-8") as empty_file:
            json.dump({}, empty_file)
        self.hyperchain = HyperChain()
        chain_is_empty = len(self.hyperchain.chain) == 0
        if chain_is_empty:
            self.hyperchain.synthesize_genesis_block()

    def tearDown(self):
        """Clean up after each test"""
        pass

    def _add_block_with_dict(self, data_dict: dict) -> Block:
        """Helper to add a block with JSON-serialized dict data"""
        json_data = json.dumps(data_dict)
        return self.hyperchain.add_block(json_data)

    def _get_block_data_dict(self, block: Block) -> dict:
        """Helper to extract dict from block's ReversibleBytes"""
        converter = ReversibleBytes("")
        decompressed = converter.readable_value(block.data.value)
        return json.loads(decompressed)

    def test_rule2_create_index_pointer_when_value_matches(self):
        """Test Rule 2: Create index pointer when incoming value matches previous block"""
        previous_block_data = {"name": "Alice", "age": 30}
        self._add_block_with_dict(previous_block_data)

        incoming_block_data = {"name": "Alice", "city": "NYC"}
        processed_data = self.hyperchain.block_data_diff(incoming_block_data)

        name_index_pointer_key = "name_<ref>"
        expected_block_index = 1
        self.assertIn(name_index_pointer_key, processed_data)
        self.assertEqual(processed_data[name_index_pointer_key], expected_block_index)
        self.assertEqual(processed_data["name"], "Alice")
        self.assertEqual(processed_data["city"], "NYC")

    def test_rule3_leave_value_unchanged_when_different(self):
        """Test Rule 3: Leave value unchanged when incoming value differs from previous"""
        previous_block_data = {"name": "Alice", "age": 30}
        self._add_block_with_dict(previous_block_data)

        incoming_block_data_with_different_name = {"name": "Bob", "age": 30}
        processed_data = self.hyperchain.block_data_diff(incoming_block_data_with_different_name)

        name_index_pointer_key = "name_<ref>"
        age_index_pointer_key = "age_<ref>"
        expected_block_index = 1
        self.assertNotIn(name_index_pointer_key, processed_data)
        self.assertEqual(processed_data["name"], "Bob")
        self.assertIn(age_index_pointer_key, processed_data)
        self.assertEqual(processed_data[age_index_pointer_key], expected_block_index)

    def test_rule1_follow_existing_index_pointer(self):
        """Test Rule 1: Follow existing index pointer to referenced block"""
        original_block_data = {"name": "Alice", "age": 30}
        self._add_block_with_dict(original_block_data)

        block_with_index_pointer = {"name": "Alice", "name_<ref>": 1, "city": "NYC"}
        self._add_block_with_dict(block_with_index_pointer)

        incoming_block_data = {"name": "Alice", "status": "active"}
        processed_data = self.hyperchain.block_data_diff(incoming_block_data)

        name_index_pointer_key = "name_<ref>"
        expected_original_block_index = 1
        self.assertIn(name_index_pointer_key, processed_data)
        self.assertEqual(processed_data[name_index_pointer_key], expected_original_block_index)
        self.assertEqual(processed_data["name"], "Alice")
        self.assertEqual(processed_data["status"], "active")

    def test_rule1_follow_pointer_then_create_new_pointer(self):
        """Test Rule 1 followed by Rule 2: Follow pointer, then create new pointer if value matches"""
        original_block_data = {"name": "Alice"}
        self._add_block_with_dict(original_block_data)

        block_with_index_pointer_to_original = {"name": "Alice", "name_<ref>": 1}
        self._add_block_with_dict(block_with_index_pointer_to_original)

        incoming_block_data_with_same_name = {"name": "Alice"}
        processed_data = self.hyperchain.block_data_diff(incoming_block_data_with_same_name)

        name_index_pointer_key = "name_<ref>"
        expected_original_block_index = 1
        self.assertIn(name_index_pointer_key, processed_data)
        self.assertEqual(processed_data[name_index_pointer_key], expected_original_block_index)
        self.assertEqual(processed_data["name"], "Alice")

    def test_multiple_keys_mixed_rules(self):
        """Test multiple keys with different rules applied"""
        initial_block_data = {"name": "Alice", "age": 30, "city": "NYC"}
        self._add_block_with_dict(initial_block_data)

        incoming_data_with_matching_name_and_age = {"name": "Alice", "age": 30, "city": "LA"}
        processed_data = self.hyperchain.block_data_diff(incoming_data_with_matching_name_and_age)

        name_index_pointer_key = "name_<ref>"
        age_index_pointer_key = "age_<ref>"
        city_index_pointer_key = "city_<ref>"
        expected_block_index = 1
        self.assertIn(name_index_pointer_key, processed_data)
        self.assertEqual(processed_data[name_index_pointer_key], expected_block_index)
        self.assertIn(age_index_pointer_key, processed_data)
        self.assertEqual(processed_data[age_index_pointer_key], expected_block_index)
        self.assertNotIn(city_index_pointer_key, processed_data)
        self.assertEqual(processed_data["city"], "LA")

    def test_empty_chain_after_genesis(self):
        """Test behavior when only genesis block exists"""
        incoming_block_data = {"name": "Alice"}
        processed_data = self.hyperchain.block_data_diff(incoming_block_data)

        name_index_pointer_key = "name_<ref>"
        self.assertEqual(processed_data, incoming_block_data)
        self.assertNotIn(name_index_pointer_key, processed_data)

    def test_key_not_in_previous_block(self):
        """Test when key doesn't exist in previous block"""
        block_with_name_only = {"name": "Alice"}
        self._add_block_with_dict(block_with_name_only)

        incoming_data_with_name_and_new_age = {"name": "Alice", "age": 30}
        processed_data = self.hyperchain.block_data_diff(incoming_data_with_name_and_new_age)

        name_index_pointer_key = "name_<ref>"
        age_index_pointer_key = "age_<ref>"
        expected_block_index = 1
        self.assertIn(name_index_pointer_key, processed_data)
        self.assertNotIn(age_index_pointer_key, processed_data)
        self.assertEqual(processed_data["age"], 30)
        self.assertEqual(processed_data[name_index_pointer_key], expected_block_index)

    def test_nested_dict_values(self):
        """Test with nested dict values"""
        block_with_nested_user_data = {"user": {"name": "Alice", "age": 30}}
        self._add_block_with_dict(block_with_nested_user_data)

        incoming_data_with_matching_nested_user = {"user": {"name": "Alice", "age": 30}}
        processed_data = self.hyperchain.block_data_diff(incoming_data_with_matching_nested_user)

        user_index_pointer_key = "user_<ref>"
        expected_block_index = 1
        self.assertIn(user_index_pointer_key, processed_data)
        self.assertEqual(processed_data[user_index_pointer_key], expected_block_index)

    def test_nested_dict_values_differ(self):
        """Test with nested dict values that differ"""
        block_with_nested_user_data = {"user": {"name": "Alice", "age": 30}}
        self._add_block_with_dict(block_with_nested_user_data)

        incoming_data_with_different_nested_user = {"user": {"name": "Bob", "age": 30}}
        processed_data = self.hyperchain.block_data_diff(incoming_data_with_different_nested_user)

        user_index_pointer_key = "user_<ref>"
        expected_different_user_data = {"name": "Bob", "age": 30}
        self.assertNotIn(user_index_pointer_key, processed_data)
        self.assertEqual(processed_data["user"], expected_different_user_data)


if __name__ == "__main__":
    unittest.main()
