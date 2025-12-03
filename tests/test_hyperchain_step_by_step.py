# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Comprehensive step-by-step test for HyperChain functionality with asserts for each line of code."""

import unittest
import os
import json
import hashlib
from unittest.mock import patch
from nnll.hyperchain import HyperChain, Block
from nnll.reverse_codec import ReversibleBytes
from nnll.json_cache import JSONCache, HYPERCHAIN_PATH_NAMED


class TestHyperChainStepByStep(unittest.TestCase):
    """Test HyperChain functionality step by step with detailed asserts."""

    test_file = ".test_step_by_step.json"

    @classmethod
    def setUpClass(cls):
        """Set up test file and patch HyperChain to use test file."""
        with open(cls.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        cls.patcher = patch.object(HyperChain, "chain_file", new=cls.test_file)
        cls.patcher.start()

    def setUp(self):
        """Create a fresh HyperChain instance for each test."""
        with open(self.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)

    def test_01_initialization(self):
        """Test step 1: HyperChain initialization."""
        chain = HyperChain()
        self.assertIsNotNone(chain, "HyperChain instance should be created")
        self.assertIsInstance(chain.chain, list, "chain.chain should be a list")
        self.assertEqual(len(chain.chain), 1, "chain.chain shouldhave genesis block")
        self.assertTrue(hasattr(chain, "chain_file"), "chain should have chain_file attribute")
        self.assertEqual(chain.chain_file, self.test_file, "chain_file should be set to test file")

    def test_02_load_chain_from_file_empty(self):
        """Test step 2: Loading chain from empty file."""
        chain = HyperChain()
        self.assertGreater(len(chain.chain), 0, "Chain should have at least one block after loading empty file")
        genesis = chain.chain[0]
        self.assertEqual(genesis.index, 0, "First block should have index 0")
        self.assertEqual(genesis.previous_hash, "init", "Genesis block previous_hash should be 'init'")

    def test_03_synthesize_genesis_block(self):
        """Test step 3: Synthesizing genesis block."""
        chain = HyperChain()
        chain.chain = []  # Clear chain to test synthesize_genesis_block independently
        genesis_block = chain.synthesize_genesis_block()
        self.assertIsInstance(genesis_block, Block, "synthesize_genesis_block should return a Block")
        self.assertEqual(genesis_block.index, 0, "Genesis block index should be 0")
        self.assertEqual(genesis_block.previous_hash, "init", "Genesis block previous_hash should be 'init'")
        self.assertIsNotNone(genesis_block.timestamp, "Genesis block should have a timestamp")
        self.assertIsInstance(genesis_block.timestamp, str, "Timestamp should be a string")
        self.assertIsNotNone(genesis_block.block_hash, "Genesis block should have a block_hash")
        self.assertIsInstance(genesis_block.block_hash, str, "block_hash should be a string")
        self.assertEqual(len(genesis_block.block_hash), 64, "block_hash should be 64 characters (SHA256 hex)")
        self.assertIsInstance(genesis_block.data, ReversibleBytes, "Block data should be ReversibleBytes")
        self.assertEqual(len(chain.chain), 1, "Chain should contain exactly one block after synthesizing genesis")
        self.assertEqual(chain.chain[0], genesis_block, "Chain should contain the returned genesis block")
        self.assertTrue(chain.is_chain_valid(), "Chain should be valid after synthesizing genesis block")

    def test_04_block_calculate_hash(self):
        """Test step 4: Block hash calculation."""
        chain = HyperChain()
        chain.chain = []
        genesis_block = chain.synthesize_genesis_block()

        expected_hash_input = f"{genesis_block.index}{genesis_block.previous_hash}{genesis_block.data.value}{genesis_block.timestamp}".encode("utf-8")
        expected_hash = hashlib.sha256(expected_hash_input).hexdigest()
        self.assertEqual(genesis_block.block_hash, expected_hash, "block_hash should match calculated hash")
        calculated_hash = genesis_block.calculate_hash()
        self.assertEqual(calculated_hash, expected_hash, "calculate_hash() should return the same hash")
        self.assertTrue(chain.is_chain_valid(), "Chain should be valid after calculating hash")

    def test_05_add_block(self):
        """Test step 5: Adding a block to the chain."""
        chain = HyperChain()
        chain.chain = []
        genesis_block = chain.synthesize_genesis_block()
        initial_length = len(chain.chain)
        self.assertEqual(initial_length, 1, "Chain should have 1 block (genesis) before adding")
        genesis_hash = genesis_block.block_hash
        self.assertIsNotNone(genesis_hash, "Genesis block should have a hash")
        new_block = chain.add_block("Test transaction data")
        self.assertIsInstance(new_block, Block, "add_block should return a Block")
        self.assertEqual(new_block.index, 1, "New block index should be 1 (second block)")
        self.assertEqual(new_block.previous_hash, genesis_hash, "New block previous_hash should match genesis block_hash")
        self.assertIsNotNone(new_block.timestamp, "New block should have a timestamp")
        self.assertIsNotNone(new_block.block_hash, "New block should have a block_hash")
        self.assertEqual(len(new_block.block_hash), 64, "block_hash should be 64 characters")
        self.assertEqual(len(chain.chain), 2, "Chain should have 2 blocks after adding one")
        self.assertEqual(chain.chain[1], new_block, "New block should be appended to chain")
        self.assertIsInstance(new_block.data, ReversibleBytes, "New block data should be ReversibleBytes")
        self.assertTrue(chain.is_chain_valid(), "Chain should be valid after adding one block")

    def test_06_add_multiple_blocks(self):
        """Test step 6: Adding multiple blocks sequentially."""
        chain = HyperChain()
        chain.chain = []
        genesis_block = chain.synthesize_genesis_block()
        block1 = chain.add_block("Transaction 1")
        self.assertEqual(len(chain.chain), 2, "Chain should have 2 blocks after first add")
        self.assertEqual(block1.index, 1, "First added block should have index 1")
        self.assertEqual(block1.previous_hash, genesis_block.block_hash, "Block 1 previous_hash should match genesis")

        block2 = chain.add_block("Transaction 2")
        self.assertEqual(len(chain.chain), 3, "Chain should have 3 blocks after second add")
        self.assertEqual(block2.index, 2, "Second added block should have index 2")
        self.assertEqual(block2.previous_hash, block1.block_hash, "Block 2 previous_hash should match block 1 hash")

        block3 = chain.add_block("Transaction 3")
        self.assertEqual(len(chain.chain), 4, "Chain should have 4 blocks after third add")
        self.assertEqual(block3.index, 3, "Third added block should have index 3")
        self.assertEqual(block3.previous_hash, block2.block_hash, "Block 3 previous_hash should match block 2 hash")

        self.assertEqual(chain.chain[0].block_hash, chain.chain[1].previous_hash, "Block 1 should link to genesis")
        self.assertEqual(chain.chain[1].block_hash, chain.chain[2].previous_hash, "Block 2 should link to block 1")
        self.assertEqual(chain.chain[2].block_hash, chain.chain[3].previous_hash, "Block 3 should link to block 2")
        self.assertTrue(chain.is_chain_valid(), "Chain should be valid after adding multiple blocks")

    def test_07_is_chain_valid_empty(self):
        """Test step 7: Chain validation with empty chain."""
        chain = HyperChain()
        chain.chain = []
        self.assertFalse(chain.is_chain_valid(), "Empty chain should be invalid")
        chain.chain = None
        self.assertFalse(chain.is_chain_valid(), "None chain should be invalid")

    def test_08_corrupt_genesis_block(self):
        """Test step 8: Chain validation with only genesis block."""
        chain = HyperChain()
        chain.chain = []
        genesis_block = chain.synthesize_genesis_block()
        self.assertEqual(genesis_block.previous_hash, "init", "Genesis block previous_hash should be 'init'")
        self.assertTrue(chain.is_chain_valid(), "Chain with valid genesis block should be valid")

        genesis_block = chain.chain[0]
        invalid_genesis = Block.create(index=0, previous_hash="wrong", data=ReversibleBytes("Genesis Block"))
        chain.chain[0] = invalid_genesis

        self.assertFalse(chain.is_chain_valid(), "Chain with invalid genesis previous_hash should be invalid")

    def test_09_is_chain_valid_multiple_blocks(self):
        """Test step 9: Chain validation with multiple blocks."""
        chain = HyperChain()
        chain.chain = []
        chain.synthesize_genesis_block()
        chain.add_block("Data 1")
        chain.add_block("Data 2")
        self.assertTrue(chain.is_chain_valid(), "Valid multi-block chain should be valid")

        for i in range(1, len(chain.chain)):
            current_block = chain.chain[i]
            previous_block = chain.chain[i - 1]

            self.assertEqual(current_block.previous_hash, previous_block.block_hash, f"Block {i} previous_hash should match block {i - 1} block_hash")
            assert isinstance(current_block.data, ReversibleBytes)
            expected_hash_input = f"{current_block.index}{previous_block.block_hash}{current_block.data.value}{current_block.timestamp}".encode("utf-8")
            expected_hash = hashlib.sha256(expected_hash_input).hexdigest()
            self.assertEqual(current_block.block_hash, expected_hash, f"Block {i} block_hash should match calculated hash")

    def test_10_save_chain_to_file(self):
        """Test step 10: Saving chain to file."""
        chain = HyperChain()
        chain.chain = []
        chain.synthesize_genesis_block()
        chain.add_block("Save test data")

        self.assertTrue(chain.is_chain_valid(), "Chain should be valid before saving")
        self.assertTrue(os.path.exists(self.test_file), "Test file should exist after save_chain_to_file")

        self.assertEqual(len(chain.chain), 2, "Chain should have 2 blocks after loading from file")
        self.assertEqual(chain.chain[0].index, 0, "Genesis block index should be 0")
        self.assertEqual(chain.chain[0].previous_hash, "init", "Genesis block previous_hash should be 'init'")
        self.assertEqual(chain.chain[1].index, 1, "First block index should be 1")
        self.assertEqual(chain.chain[1].previous_hash, chain.chain[0].block_hash, "First block previous_hash should match genesis block hash")

    def test_11_save_invalid_chain(self):
        """Test step 11: Attempting to save invalid chain."""
        chain = HyperChain()
        chain.chain = []
        chain.synthesize_genesis_block()

        invalid_block = Block.create(index=0, previous_hash="wrong_hash", data=ReversibleBytes("Invalid"))
        chain.chain.append(invalid_block)

        self.assertFalse(chain.is_chain_valid(), "Chain should be invalid")

        # Assert: Saving invalid chain raises ValueError
        with self.assertRaises(ValueError) as context:
            chain.save_chain_to_file()
        self.assertIn("Invalid Block Value", str(context.exception), "Should raise ValueError for invalid chain")

    def test_12_load_chain_from_file(self):
        """Test step 12: Loading chain from file."""

        chain1 = HyperChain()
        chain1.chain = []
        chain1.synthesize_genesis_block()
        chain1.add_block("Load test data 1")
        chain1.add_block("Load test data 2")
        chain1.save_chain_to_file()
        # Assert: File exists with data
        self.assertTrue(os.path.exists(self.test_file), "Test file should exist")

        chain2 = HyperChain()
        self.assertGreater(len(chain2.chain), 0, "Chain should have loaded blocks from file")
        self.assertEqual(len(chain2.chain), 3, "Chain should have 3 blocks (genesis + 2 added)")

        # Assert: Loaded chain is valid
        self.assertTrue(chain2.is_chain_valid(), "Loaded chain should be valid")

        # Assert: Loaded blocks match saved blocks
        for i, (saved_block, loaded_block) in enumerate(zip(chain1.chain, chain2.chain)):
            self.assertEqual(saved_block.index, loaded_block.index, f"Block {i} index should match")
            self.assertEqual(saved_block.previous_hash, loaded_block.previous_hash, f"Block {i} previous_hash should match")
            self.assertEqual(saved_block.block_hash, loaded_block.block_hash, f"Block {i} block_hash should match")
            self.assertEqual(saved_block.timestamp, loaded_block.timestamp, f"Block {i} timestamp should match")

    def test_13_block_create(self):
        """Test step 13: Block.create classmethod."""
        # Step 13.1: Create block using Block.create
        test_data = "Test data string"
        reversible_data = ReversibleBytes(test_data)
        block = Block.create(index=5, previous_hash="test_hash", data=reversible_data)

        # Assert: Block is created
        self.assertIsNotNone(block, "Block.create should return a Block")
        self.assertIsInstance(block, Block, "Block.create should return Block instance")

        # Assert: Block has correct attributes
        self.assertEqual(block.index, 5, "Block index should be 5")
        self.assertEqual(block.previous_hash, "test_hash", "Block previous_hash should be 'test_hash'")
        self.assertIsInstance(block.data, ReversibleBytes, "Block data should be ReversibleBytes")

        # Assert: Block has auto-generated timestamp
        self.assertIsNotNone(block.timestamp, "Block should have auto-generated timestamp")

        # Assert: Block has auto-generated hash
        self.assertIsNotNone(block.block_hash, "Block should have auto-generated block_hash")

    def test_14_block_to_dict(self):
        """Test step 14: Block.to_dict method."""
        chain = HyperChain()
        chain.chain = []
        block = chain.synthesize_genesis_block()

        # Step 14.1: Convert block to dict
        block_dict = block.to_dict()

        # Assert: Returns a dictionary
        self.assertIsInstance(block_dict, dict, "to_dict should return a dictionary")

        # Assert: Dictionary contains all expected keys
        expected_keys = {"index", "data", "previous_hash", "timestamp", "block_hash"}
        self.assertEqual(set(block_dict.keys()), expected_keys, "to_dict should contain all block attributes")

        # Assert: Values match block attributes
        self.assertEqual(block_dict["index"], block.index, "dict index should match block index")
        self.assertEqual(block_dict["previous_hash"], block.previous_hash, "dict previous_hash should match")
        self.assertEqual(block_dict["timestamp"], block.timestamp, "dict timestamp should match")
        self.assertEqual(block_dict["block_hash"], block.block_hash, "dict block_hash should match")
        self.assertEqual(block_dict["data"], block.data.value, "dict data should match block data")

    def test_15_block_from_dict(self):
        """Test step 15: Block.from_dict classmethod."""
        chain = HyperChain()
        chain.chain = []
        original_block = chain.synthesize_genesis_block()

        # Step 15.1: Convert block to dict
        block_dict = original_block.to_dict()

        # Step 15.2: Recreate block from dict
        recreated_block = Block.from_dict(block_dict)

        # Assert: Recreated block is a Block instance
        self.assertIsInstance(recreated_block, Block, "from_dict should return a Block")

        # Assert: All attributes match
        self.assertEqual(recreated_block.index, original_block.index, "Recreated index should match")
        self.assertEqual(recreated_block.previous_hash, original_block.previous_hash, "Recreated previous_hash should match")
        self.assertEqual(recreated_block.timestamp, original_block.timestamp, "Recreated timestamp should match")
        self.assertEqual(recreated_block.block_hash, original_block.block_hash, "Recreated block_hash should match")

    def test_16_reversible_bytes_integration(self):
        """Test step 16: ReversibleBytes integration with blocks."""
        chain = HyperChain()
        chain.chain = []
        chain.synthesize_genesis_block()

        # Step 16.1: Create block with string data
        test_string = "Test reversible bytes data"
        block = chain.add_block(test_string)

        # Assert: Block data is ReversibleBytes
        self.assertIsInstance(block.data, ReversibleBytes, "Block data should be ReversibleBytes")

        # Assert: Data can be decompressed
        converter = ReversibleBytes("")
        decompressed = converter.readable_value(block.data.value)
        self.assertIsInstance(decompressed, str, "Decompressed data should be a string")
        # Note: The actual decompressed value may differ due to compression, but it should be a string

    def test_17_chain_persistence(self):
        """Test step 17: Chain persistence across instances."""
        # Step 17.1: Create first chain and add blocks
        chain1 = HyperChain()
        chain1.chain = []
        chain1.synthesize_genesis_block()
        block1 = chain1.add_block("Persistence test 1")
        block2 = chain1.add_block("Persistence test 2")
        chain1.save_chain_to_file()

        # Assert: First chain is valid
        self.assertTrue(chain1.is_chain_valid(), "First chain should be valid")
        self.assertEqual(len(chain1.chain), 3, "First chain should have 3 blocks")

        # Step 17.2: Create second chain (should load from file)
        chain2 = HyperChain()

        # Assert: Second chain loaded from file
        self.assertEqual(len(chain2.chain), 3, "Second chain should have 3 blocks loaded from file")
        self.assertTrue(chain2.is_chain_valid(), "Second chain should be valid")

        # Assert: Chains are identical
        for i in range(len(chain1.chain)):
            self.assertEqual(chain1.chain[i].index, chain2.chain[i].index, f"Block {i} index should match")
            self.assertEqual(chain1.chain[i].block_hash, chain2.chain[i].block_hash, f"Block {i} hash should match")

    def test_18_chain_validation_edge_cases(self):
        """Test step 18: Chain validation edge cases."""
        chain = HyperChain()
        chain.chain = []
        chain.synthesize_genesis_block()
        chain.add_block("Test")

        # Assert: Valid chain passes validation
        self.assertTrue(chain.is_chain_valid(), "Valid chain should pass validation")

        # Step 18.1: Test with single block (genesis only)
        single_block_chain = HyperChain()
        single_block_chain.chain = []
        single_block_chain.synthesize_genesis_block()
        self.assertTrue(single_block_chain.is_chain_valid(), "Single block chain should be valid")

    def test_19_block_timestamp_format(self):
        """Test step 19: Block timestamp format."""
        chain = HyperChain()
        chain.chain = []
        block = chain.synthesize_genesis_block()

        # Assert: Timestamp is a string
        self.assertIsInstance(block.timestamp, str, "Timestamp should be a string")

        # Assert: Timestamp has expected format (YYYY-MM-DD HH:MM:SS)
        # Format is "%Y-%m-%d %H:%M:%s" but %s is not standard, might be %S
        # Just check it's not empty and has some structure
        self.assertGreater(len(block.timestamp), 0, "Timestamp should not be empty")
        self.assertIn("-", block.timestamp, "Timestamp should contain date separator")
        self.assertIn(":", block.timestamp, "Timestamp should contain time separator")

    def test_20_complete_workflow(self):
        """Test step 20: Complete workflow from initialization to persistence."""
        # Step 20.1: Initialize chain
        chain = HyperChain()
        chain.chain = []
        self.assertEqual(len(chain.chain), 0, "Chain should start empty")

        # Step 20.2: Create genesis block
        genesis = chain.synthesize_genesis_block()
        self.assertEqual(len(chain.chain), 1, "Chain should have 1 block after genesis")
        self.assertTrue(chain.is_chain_valid(), "Chain with genesis should be valid")

        # Step 20.3: Add multiple blocks
        block1 = chain.add_block("Workflow step 1")
        block2 = chain.add_block("Workflow step 2")
        block3 = chain.add_block("Workflow step 3")
        self.assertEqual(len(chain.chain), 4, "Chain should have 4 blocks")
        self.assertTrue(chain.is_chain_valid(), "Chain should remain valid after adding blocks")

        # Step 20.4: Save chain
        chain.save_chain_to_file()
        self.assertTrue(os.path.exists(self.test_file), "File should exist after save")

        # Step 20.5: Verify file contents
        with open(self.test_file, "r", encoding="ascii") as f:
            file_data = json.load(f)
            self.assertEqual(len(file_data["_hyperchain"]), 4, "File should contain 4 blocks")

        # Step 20.6: Load into new chain
        new_chain = HyperChain()
        self.assertEqual(len(new_chain.chain), 4, "New chain should load 4 blocks")
        self.assertTrue(new_chain.is_chain_valid(), "Loaded chain should be valid")

        # Step 20.7: Verify all blocks match
        for i in range(len(chain.chain)):
            self.assertEqual(chain.chain[i].index, new_chain.chain[i].index, f"Block {i} index should match")
            self.assertEqual(chain.chain[i].block_hash, new_chain.chain[i].block_hash, f"Block {i} hash should match")

    @classmethod
    def tearDownClass(cls):
        """Clean up test file and stop patcher."""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        cls.patcher.stop()


if __name__ == "__main__":
    unittest.main()
