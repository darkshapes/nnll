# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""test"""

# pylint:disable=line-too-long
import unittest
from unittest.mock import MagicMock, patch
import hashlib
from nnll.integrity.hyper_chain import HyperChain
from nnll.mir.json_cache import JSONCache
import json


class TestHyperChain(unittest.TestCase):
    """linked-list test"""

    test_file = ".test.json"

    @classmethod
    def setUpClass(cls):
        with open(cls.test_file, "w", encoding="UTF-8") as doc:
            json.dump({}, doc)
        cls.patcher = patch.object(HyperChain, "chain_file", new=JSONCache(cls.test_file))
        cls.patcher.start()
        cls.hyperchain = HyperChain()

    def test_chain_origin(self):
        """simulate new chain"""
        self.hyperchain.synthesize_genesis_block()

        self.hyperchain.add_block("Transaction 1")
        self.hyperchain.add_block("Transaction 2")

        self.assert_block_tags(index=0, previous_hash="init", data="Genesis Block")
        self.assert_block_tags(index=1, previous_hash=self.hyperchain.chain[0].block_hash, data="Transaction 1")
        self.assert_block_tags(index=2, previous_hash=self.hyperchain.chain[1].block_hash, data="Transaction 2")
        # print(self.hyperchain.chain)
        self.assertTrue(self.hyperchain.is_chain_valid())

    def assert_block_tags(self, **kwargs):
        """loop for block attributes"""
        # print(kwargs)
        for k, v in kwargs.items():
            # print(getattr(self.hyperchain.chain[kwargs["index"]], f"{k}"))
            assert getattr(self.hyperchain.chain[kwargs["index"]], f"{k}") == v

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
        cls.patcher = patch.object(HyperChain, "chain_file", new=JSONCache(cls.test_file))
        cls.patcher.start()
        cls.hyperchain = HyperChain()
        cls.genesis_block = MagicMock(
            index=0,
            previous_hash="init",
            data="Genesis Data",
            timestamp="timestamp1",
            block_hash="genesis_hash",
        )
        cls.block1 = MagicMock(
            index=1,
            previous_hash=cls.genesis_block.block_hash,
            data="Block 1 Data",
            timestamp="timestamp2",
            block_hash=hashlib.sha256("1genesis_hashBlock 1 Datatimestamp2".encode("utf-8")).hexdigest(),
        )
        cls.block2 = MagicMock(
            index=2,
            previous_hash=cls.block1.block_hash,
            data="Block 2 Data",
            timestamp="timestamp3",
            block_hash=hashlib.sha256(f"2{cls.block1.block_hash}Block 2 Datatimestamp3".encode("utf-8")).hexdigest(),
        )

        cls.hyperchain.chain = [cls.genesis_block, cls.block1, cls.block2]

    def test_valid_chain(self):
        """Test validator against
        correct chain, invalid genesis block, tampered block, invalid hash
        """
        self.assertTrue(self.hyperchain.is_chain_valid())

        self.genesis_block.previous_hash = "invalid"
        self.assertFalse(self.hyperchain.is_chain_valid())

        self.block1.data = "Tampered Data"

        with unittest.mock.patch("hashlib.sha256") as mock_sha:
            mock_sha.return_value.hexdigest.return_value = "tampered_hash"
            self.assertFalse(self.hyperchain.is_chain_valid())

        #
        self.block2.previous_hash = "wrong_previous_hash"
        self.assertFalse(self.hyperchain.is_chain_valid())

    @classmethod
    def tearDownClass(cls):
        import os

        os.remove(cls.test_file)
        cls.patcher.stop()
