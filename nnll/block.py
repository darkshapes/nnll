# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Basic block class"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass
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

        return strftime("%Y-%m-%d %H:%M:%S", gmtime(time_ns() // 1e9))

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
