# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint:disable=line-too-long

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError
from nnll.block import Block
from nnll.reverse_codec import ReversibleBytes


class TestBlock(unittest.TestCase):
    """Test blocks objects"""

    test_file = ".test.json"

    def setUp(self):
        """Create block and save existing parameters"""
        with open(self.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        self.block = Block.create(index=1, previous_hash="previous_hash_0", data=ReversibleBytes("Sample Data"))

        # Use to_dict() instead of asdict() for consistency with Block's serialization
        self.original_block_dict = self.block.to_dict()

    def test_block_creation(self):
        """Validate block creation. Ensure that the block is immutable. Check attributes"""
        with self.assertRaises(FrozenInstanceError):
            self.block.index = 2

        self.assertIsInstance(self.block.index, int)
        self.assertIsInstance(self.block.previous_hash, str)
        self.assertIsInstance(self.block.data, ReversibleBytes)
        self.assertIsInstance(self.block.timestamp, str)
        self.assertIsInstance(self.block.block_hash, str)

    def test_block_attributes(self):
        """Validate that block attribute properties remain consistent."""
        for attrib_name, expected_value in self.original_block_dict.items():
            actual_value = getattr(self.block, attrib_name)

            if attrib_name == "data":
                # data is a ReversibleBytes object, compare the compressed value
                self.assertIsInstance(actual_value, ReversibleBytes, "Block data should be ReversibleBytes")
                self.assertEqual(actual_value.value, expected_value, "Block data.value should match stored value")
            else:
                self.assertEqual(actual_value, expected_value, f"Block {attrib_name} should match stored value")

    def test_block_hash_validity(self):
        """Validate the block hash calculation by recalculating and comparison."""
        # Use the same format as calculate_hash() - data.value (compressed string)
        expected_hash = hashlib.sha256(f"{self.block.index}{self.block.previous_hash}{self.block.data.value}{self.block.timestamp}".encode("utf-8")).hexdigest()

        self.assertEqual(self.block.block_hash, expected_hash)

    def test_block_to_dict_from_dict(self):
        """Test serialization and deserialization of blocks."""
        # Serialize block
        block_dict = self.block.to_dict()

        # Verify data is stored as compressed string
        self.assertIsInstance(block_dict["data"], str, "Serialized data should be a string")

        # Deserialize block
        recreated_block = Block.from_dict(block_dict)

        # Verify all attributes match
        self.assertEqual(self.block.index, recreated_block.index)
        self.assertEqual(self.block.previous_hash, recreated_block.previous_hash)
        self.assertEqual(self.block.timestamp, recreated_block.timestamp)
        self.assertEqual(self.block.block_hash, recreated_block.block_hash)

        # Verify data can be decompressed to original
        converter = ReversibleBytes("")
        original_text = converter.readable_value(self.block.data.value)
        recreated_text = converter.readable_value(recreated_block.data.value)
        self.assertEqual(original_text, recreated_text, "Decompressed data should match original")

    def tearDown(self):
        import os

        os.remove(self.test_file)


if __name__ == "__main__":
    unittest.main()
