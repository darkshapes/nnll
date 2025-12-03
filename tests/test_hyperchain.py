# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""test"""

# pylint:disable=line-too-long
import unittest
from unittest.mock import MagicMock, patch
import hashlib
from nnll.hyperchain import HyperChain
from nnll.reverse_codec import ReversibleBytes
import json


class TestHyperChain(unittest.TestCase):
    """linked-list test"""

    test_file = ".test.json"

    @classmethod
    def setUpClass(cls):
        with open(cls.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        cls.patcher = patch.object(HyperChain, "chain_file", new=cls.test_file)
        cls.patcher.start()
        cls.hyperchain = HyperChain()

    def test_chain_origin(self):
        """simulate new chain"""
        self.hyperchain.chain = []  # Clear chain to start fresh
        self.hyperchain.synthesize_genesis_block()

        self.hyperchain.add_block("Transaction 1")
        self.hyperchain.add_block("Transaction 2")
        print(self.hyperchain.chain)

        # Check genesis block
        genesis = self.hyperchain.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.previous_hash, "init")
        self.assertIsInstance(genesis.data, ReversibleBytes)
        # Verify decompressed data matches original
        converter = ReversibleBytes("")
        decompressed = converter.readable_value(genesis.data.value)
        self.assertEqual(decompressed, "Genesis Block")

        # Check block 1
        block1 = self.hyperchain.chain[1]
        self.assertEqual(block1.index, 1)
        self.assertEqual(block1.previous_hash, genesis.block_hash)
        self.assertIsInstance(block1.data, ReversibleBytes)
        decompressed = converter.readable_value(block1.data.value)
        self.assertEqual(decompressed, "Transaction 1")

        # Check block 2
        block2 = self.hyperchain.chain[2]
        self.assertEqual(block2.index, 2)
        self.assertEqual(block2.previous_hash, block1.block_hash)
        self.assertIsInstance(block2.data, ReversibleBytes)
        decompressed = converter.readable_value(block2.data.value)
        self.assertEqual(decompressed, "Transaction 2")

        self.assertTrue(self.hyperchain.is_chain_valid())

    @classmethod
    def tearDownClass(cls):
        import os

        os.remove(cls.test_file)
        cls.patcher.stop()


class TestHyperChainValidation(unittest.TestCase):
    """validation tester"""

    test_file = ".test.json"

    @classmethod
    def setUpClass(cls):
        with open(cls.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        cls.patcher = patch.object(HyperChain, "chain_file", new=cls.test_file)
        cls.patcher.start()
        cls.hyperchain = HyperChain()

        # Create ReversibleBytes objects for mock data
        genesis_data = ReversibleBytes("Genesis Data")
        block1_data = ReversibleBytes("Block 1 Data")
        block2_data = ReversibleBytes("Block 2 Data")

        # Calculate correct hashes using data.value (compressed string)
        block1_hash_input = f"1genesis_hash{block1_data.value}timestamp2".encode("utf-8")
        block1_hash = hashlib.sha256(block1_hash_input).hexdigest()

        block2_hash_input = f"2{block1_hash}{block2_data.value}timestamp3".encode("utf-8")
        block2_hash = hashlib.sha256(block2_hash_input).hexdigest()

        cls.genesis_block = MagicMock(
            index=0,
            previous_hash="init",
            data=genesis_data,
            timestamp="timestamp1",
            block_hash="genesis_hash",
        )
        cls.block1 = MagicMock(
            index=1,
            previous_hash=cls.genesis_block.block_hash,
            data=block1_data,
            timestamp="timestamp2",
            block_hash=block1_hash,
        )
        cls.block2 = MagicMock(
            index=2,
            previous_hash=cls.block1.block_hash,
            data=block2_data,
            timestamp="timestamp3",
            block_hash=block2_hash,
        )

        cls.hyperchain.chain = [cls.genesis_block, cls.block1, cls.block2]

    def test_valid_chain(self):
        """Test validator against
        correct chain, invalid genesis block, tampered block, invalid hash
        """
        self.assertTrue(self.hyperchain.is_chain_valid())

        self.genesis_block.previous_hash = "invalid"
        self.assertFalse(self.hyperchain.is_chain_valid())

        # Reset for next test
        self.genesis_block.previous_hash = "init"

        # Tamper with block data - create new ReversibleBytes with different value
        tampered_data = ReversibleBytes("Tampered Data")
        self.block1.data = tampered_data
        # Hash will no longer match because data.value changed
        self.assertFalse(self.hyperchain.is_chain_valid())

        # Reset for next test
        self.block1.data = ReversibleBytes("Block 1 Data")
        # Recalculate hash to match
        block1_hash_input = f"1genesis_hash{self.block1.data.value}timestamp2".encode("utf-8")
        self.block1.block_hash = hashlib.sha256(block1_hash_input).hexdigest()

        # Test invalid previous_hash link
        self.block2.previous_hash = "wrong_previous_hash"
        self.assertFalse(self.hyperchain.is_chain_valid())

    @classmethod
    def tearDownClass(cls):
        import os

        os.remove(cls.test_file)
        cls.patcher.stop()
