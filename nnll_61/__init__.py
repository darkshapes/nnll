### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


"""區塊鏈儲存 Blockchain storage"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass

from nnll_60 import CHAIN_PATH_NAMED, JSONCache


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
    data: str
    timestamp: str = None
    block_hash: str = None

    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", self.create_timestamp())
        if self.block_hash is None:
            object.__setattr__(self, "block_hash", self.calculate_hash())

    def calculate_hash(self) -> str:
        import hashlib

        block_contents = f"{self.index}{self.previous_hash}{self.data}{self.timestamp}".encode("utf-8")
        return hashlib.sha256(block_contents).hexdigest()

    def create_timestamp(self):
        from time import gmtime, strftime, time_ns

        return strftime("%Y-%m-%d %H:%M:%s", gmtime(time_ns() // 1e9))

    @classmethod
    def create(cls, index: int, previous_hash: str, data: str) -> "Block":
        return cls(index=index, previous_hash=previous_hash, data=data)

    @classmethod
    def from_dict(cls, data: dict):
        block = cls(index=data["index"], data=data["data"], previous_hash=data["previous_hash"])
        object.__setattr__(block, "timestamp", data["timestamp"])
        object.__setattr__(block, "block_hash", data["block_hash"])
        return block

    def to_dict(self):
        return {
            "index": self.index,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "block_hash": self.block_hash,
        }


class HyperChain:
    """
    Chain operations to blocks.\n
    Use sequence of hashes to assert validity of links.\n
    """

    chain_file = JSONCache(CHAIN_PATH_NAMED)

    @chain_file.decorator
    def __init__(self, data=None):
        self.chain = []
        self.chain_file.refresh()
        if "_hyperchain" in data:
            self.load_chain_from_file(data)
            # Ensure latest data is loaded
        # self.cache = chain_file

    def synthesize_genesis_block(self) -> Block:
        """Runs once."""
        genesis_block = Block.create(index=0, previous_hash="init", data="Genesis Block")
        self.chain.append(genesis_block)
        return genesis_block

    def add_block(self, data: str) -> Block:
        """
        Add a new block to the chain\n
        :param data: The contents to store on-chain
        :type data: str
        :return: `Block` the new block
        """
        index = len(self.chain)
        # print(self.chain)
        previous_hash = self.chain[-1].block_hash
        new_block = Block.create(index=index, previous_hash=previous_hash, data=data)
        self.chain.append(new_block)
        self.save_chain_to_file()
        return new_block

    @chain_file.decorator
    def save_chain_to_file(self, data: str = None):  # pylint:disable=unused-argument
        """Will not save unless chain is valid"""
        from dataclasses import asdict

        if self.is_chain_valid():
            self.chain_file.update_cache({"_hyperchain": [asdict(block) for block in self.chain]})
        else:
            raise ValueError("Invalid Block Value, chain cannot save. File handling error or otherwise corrupt chain.")

    # @chain_file.decorator
    def load_chain_from_file(self, data: str):
        """Load and validate chain"""
        for block_data in data["_hyperchain"]:
            block = Block.from_dict(block_data)
            self.chain.append(block)
        if not self.is_chain_valid():
            self.chain = []
            raise ValueError("Invalid Block Value, chain cannot load. File handling error or otherwise corrupt chain.")

    def is_chain_valid(self) -> bool:
        """
        Hash validation by recompute and comparison for blocks and links.\n
        Validate genesis block separately.
        """
        import hashlib

        if self.chain[0].previous_hash != "init":
            return False

        # Loop through each block starting from index 1 (the second block).
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            current_tag = f"{current_block.index}{previous_block.block_hash}{current_block.data}{current_block.timestamp}"
            current_tag = current_tag.encode("utf-8")
            # print(current_block.block_hash, hashlib.sha256(current_tag).hexdigest())
            if not current_block.block_hash == hashlib.sha256(current_tag).hexdigest():
                return False

            # print((current_block.previous_hash, previous_block.block_hash))
            if not current_block.previous_hash == previous_block.block_hash:
                return False

        return True
