# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


"""區塊鏈儲存 Linked-list storage"""

# pylint:disable=line-too-long, import-outside-toplevel

import json
import os
from nnll.json_cache import HYPERCHAIN_PATH_NAMED
from nnll.reverse_codec import ReversibleBytes
from nnll.block import Block


class HyperChain:
    """
    Chain operations to blocks.\n
    Use sequence of hashes to assert validity of links.\n
    """

    chain_file = HYPERCHAIN_PATH_NAMED

    def __init__(self, data=chain_file):
        self.chain = []
        self.load_chain_from_file()

    def synthesize_genesis_block(self) -> Block:
        """Runs once."""
        genesis_data = ReversibleBytes("Genesis Block")
        genesis_block = Block.create(index=0, previous_hash="init", data=genesis_data)
        self.chain.append(genesis_block)
        return genesis_block

    def add_block(self, data: str) -> Block:
        """
        Add a new block to the chain\n
        :param data: The contents to store on-chain
        :return: `Block` the new block
        """
        index = len(self.chain)
        previous_hash = self.chain[-1].block_hash
        reversible_bytes = ReversibleBytes(data)

        new_block = Block.create(index=index, previous_hash=previous_hash, data=reversible_bytes)
        self.chain.append(new_block)
        self.save_chain_to_file()
        return new_block

    def save_chain_to_file(self):  # pylint:disable=unused-argument
        """Will not save unless chain is valid"""

        if self.is_chain_valid():
            with open(str(self.chain_file), "w", encoding="ascii") as doc:
                json.dump(obj={"_hyperchain": [block.to_dict() for block in self.chain]}, fp=doc, ensure_ascii=False, indent=4)
        else:
            raise ValueError("Invalid Block Value, chain cannot save. File handling error or otherwise corrupt chain.")

    def load_chain_from_file(self):
        """Load and validate chain"""
        from nnll.helpers import ensure_path

        ensure_path(os.path.dirname(self.chain_file), os.path.basename(self.chain_file))
        try:
            with open(str(self.chain_file), "r", encoding="ascii") as doc:
                content = doc.read()
                if content:
                    data = json.loads(content)
                    if not data.get("_hyperchain"):
                        self.synthesize_genesis_block()
                        self.save_chain_to_file()
                    else:
                        for block_data in data.get("_hyperchain"):
                            block = Block.from_dict(block_data)
                            self.chain.append(block)
                else:
                    # File is empty, create genesis block
                    self.synthesize_genesis_block()
                    self.save_chain_to_file()
        except (FileNotFoundError, json.JSONDecodeError):
            # File doesn't exist or is invalid, create genesis block
            self.synthesize_genesis_block()
            self.save_chain_to_file()

        if not self.is_chain_valid():
            print(self.chain)
            self.chain = []
            raise ValueError("Invalid Block Value, chain cannot load. File handling error or otherwise corrupt chain.")

    def is_chain_valid(self) -> bool:
        """
        Hash validation by recompute and comparison for blocks and links.\n
        Validate genesis block separately.
        """
        import hashlib

        if self.chain is None or len(self.chain) == 0:
            return False
        if self.chain[0].previous_hash != "init":
            return False

        # Loop through each block starting from index 1 (the second block).
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            current_tag = f"{current_block.index}{previous_block.block_hash}{current_block.data.value}{current_block.timestamp}"
            current_tag = current_tag.encode("utf-8")
            if not current_block.block_hash == hashlib.sha256(current_tag).hexdigest():
                return False

            if not current_block.previous_hash == previous_block.block_hash:
                return False

        return True

    def block_data_diff(self, incoming_data: dict) -> dict:
        """Add new data applying the three rules. \n
        When previous k/v pair is index, **copy indexref**. \n
        When previous k/v pair equals new, **create index ref**. \n
        When previous k/v pair differs, move to next k/v pair. \n
        :param incoming_data: The new data to add to the chain
        :return: Mapping with index pointers added, or original data if new data is identical
        """
        converter = ReversibleBytes("")

        def _get_block_data_dict(block: Block) -> dict:
            """Extract dict from block's ReversibleBytes"""
            try:
                decompressed = converter.readable_value(block.data.value)
                return json.loads(decompressed)
            except (json.JSONDecodeError, ValueError):
                # If data is not JSON, return empty dict
                return {}

        mutable_data = incoming_data.copy()  # create a copy of the new data to avoid modifying the original data

        for key, _ in incoming_data.items():  # iterate through the new data
            # Reset state for each key - start with the last block
            previous_data_index = len(self.chain) - 1
            index_block = self.chain[-1]
            index_pointer_key: str = f"{key}_<ref>"  # the unique key for this key's index pointer
            block_data: dict = _get_block_data_dict(index_block)  # get the previous data

            if block_data.get(index_pointer_key, None):  # rule1 – previous entry is an index pointer, follow it
                previous_data_index = block_data[index_pointer_key]  # get the previous index
                index_block = self.chain[previous_data_index]  # get the block at the previous index
                block_data = _get_block_data_dict(index_block)  # get the previous data

            if incoming_data[key] == block_data.get(key, {}):  # if the new data is the same as the previous data,
                mutable_data[index_pointer_key] = previous_data_index  # create an index pointer (rule 2)
            # if the new data is different, we leave the value unchanged (rule 3)

        return mutable_data
