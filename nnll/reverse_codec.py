# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Reverse codec for compressing and decompressing data"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass
import zlib
import base64


@dataclass
class ReversibleBytes:
    value: str

    def __post_init__(self):
        if self.value is not None:
            self.value = self._compress_state(self.value)

    def __getstate__(self):
        return {"value": self.value}

    def __setstate__(self, state):
        self.value = state["value"]

    @classmethod
    def _compress_state(cls, text: str) -> "ReversibleBytes":
        """Compress plain text and reverse the encoded string\n
        :param plain_text: The plain text to compress
        :return: Compressed transformed text"""
        data = text.encode("utf-8")
        data = cls.z_compress(data)
        data = cls.b85_encode(data)
        return data

    def readable_value(self, data: str) -> str:
        """Decompress the encoded string and reverse the decoded string"""
        # Reverse the compression steps in reverse order
        data_bytes = self.b85_decode(data)  # 1. str → bytes
        data_bytes = self.z_decompress(data_bytes)  # 3. undo compress
        original_text = data_bytes.decode("utf-8")  # 6. bytes → original str
        return original_text

    @classmethod
    def z_compress(cls, x: bytes) -> bytes:
        return zlib.compress(x)

    @classmethod
    def z_decompress(cls, x: bytes) -> bytes:
        return zlib.decompress(x)

    @classmethod
    def b85_encode(cls, x: bytes) -> str:
        return base64.b85encode(x).decode("ascii")

    @classmethod
    def b85_decode(cls, x: str) -> bytes:
        return base64.b85decode(x)
