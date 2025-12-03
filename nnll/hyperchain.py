# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


"""區塊鏈儲存 Linked-list storage"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass
import json
from nnll.json_cache import JSONCache, HYPERCHAIN_PATH_NAMED
from nnll.reverse_codec import ReversibleBytes


@dataclass(frozen=True)
class Block:
    """
    Basic block class.\n
    Use hash value to assert validity of contents\n
    Frozen creates immutability.\n
    Attributes are locked after creation.\n
    Warning: Clock precision may contain OS specific implementation differences.\n

    Block schema\n
    [index_num], [timestamp], [prior_hash], [contents], [hash]\n
    `int`, `YYYY-MM-DD` `HH:MM:SSSSSSSSSS`, 0a0a0a0a0, <data>, "9f9f9f9f9"\n
    """

    index: int
    previous_hash: str
    data: ReversibleBytes
    timestamp: str = None
    block_hash: str = None

    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", self.create_timestamp())
        if self.block_hash is None:
            object.__setattr__(self, "block_hash", self.calculate_hash())

    def calculate_hash(self) -> str:
        """Calculate hash for string of prior hash, contents, and time combined"""
        import hashlib

        block_contents = f"{self.index}{self.previous_hash}{self.data.value}{self.timestamp}".encode("utf-8")
        return hashlib.sha256(block_contents).hexdigest()

    def create_timestamp(self):
        """Add timestamp attribute"""
        from time import gmtime, strftime, time_ns

        return strftime("%Y-%m-%d %H:%M:%s", gmtime(time_ns() // 1e9))

    @classmethod
    def create(cls, index: int, previous_hash: str, data: ReversibleBytes) -> "Block":
        """Form a new block"""
        return cls(index=index, previous_hash=previous_hash, data=data)

    @classmethod
    def from_dict(cls, stored_data: dict):
        """Recreate existing block"""
        converter = ReversibleBytes("")
        decompressed_text = converter.readable_value(stored_data["data"])  # Returns string
        readable_bytes = ReversibleBytes(decompressed_text)  # Create new ReversibleBytes from decompressed text
        block = cls(index=stored_data["index"], data=readable_bytes, previous_hash=stored_data["previous_hash"])
        object.__setattr__(block, "timestamp", stored_data["timestamp"])
        object.__setattr__(block, "block_hash", stored_data["block_hash"])
        return block

    def to_dict(self):
        """Flatten block into serializable structure"""
        return {
            "index": self.index,
            "data": self.data.value,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "block_hash": self.block_hash,
        }


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

        ensure_path(self.chain_file)
        with open(str(self.chain_file), "r", encoding="ascii") as doc:
            if content := doc.read():
                data = json.loads(content)
                if not data.get("_hyperchain"):
                    self.synthesize_genesis_block()
                    self.save_chain_to_file()
                else:
                    for block_data in data.get("_hyperchain"):
                        block = Block.from_dict(block_data)
                        self.chain.append(block)
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
