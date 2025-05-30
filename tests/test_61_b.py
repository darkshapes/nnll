### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint:disable=line-too-long


import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, asdict
from nnll_61 import Block


class TestBlock(unittest.TestCase):
    """Test blocks objects"""

    test_file = ".test.json"
    file_name = ".test.json"

    def setUp(self):
        """Create block and save existing parameters"""
        with open(self.file_name, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        self.block = Block.create(index=1, previous_hash="previous_hash_0", data="Sample Data")

        self.original_block_attribs = dict(asdict(self.block).items())

    def test_block_creation(self):
        """Validate block creation. Ensure that the block is immutable. Check attributes"""
        with self.assertRaises(FrozenInstanceError):
            self.block.index = 2

        self.assertIsInstance(self.block.index, int)
        self.assertIsInstance(self.block.previous_hash, str)
        self.assertIsInstance(self.block.data, str)
        self.assertIsInstance(self.block.timestamp, str)
        self.assertIsInstance(self.block.block_hash, str)

    def test_block_attributes(self):
        """Validate that block attribute properties remain consistent."""
        for attrib_name, value in self.original_block_attribs.items():
            self.assertEqual(getattr(self.block, attrib_name), value)

    def test_block_hash_validity(self):
        """Validate the block hash calculation by recalcuating and comparison."""
        expected_hash = hashlib.sha256(f"{self.block.index}{self.block.previous_hash}{self.block.data}{self.block.timestamp}".encode("utf-8")).hexdigest()

        self.assertEqual(self.block.block_hash, expected_hash)


if __name__ == "__main__":
    unittest.main()
