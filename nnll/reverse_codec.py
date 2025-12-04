# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Reverse codec for compressing and decompressing data"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass
import zlib
import base64
import msgpack


@dataclass
class ReversibleBytes:
    value: str

    def __post_init__(self) -> None:
        if self.value is not None:
            self.__setstate__({"value": self._compress_state(self.value)})

    def __getstate__(self) -> dict[str, str]:
        return {"value": self.value}

    def __setstate__(self, state) -> None:
        self.value = state["value"]

    @classmethod
    def _compress_state(cls, text: str) -> str:
        """Compress plain text and reverse the encoded string\n
        :param plain_text: The plain text to compress
        :return: Compressed transformed text"""
        unicode_value = text.encode("utf-8")
        serialized_bytes = msgpack.packb(unicode_value)
        compressed_bytes: bytes = cls.z_compress(serialized_bytes)  # type: ignore[arg-type]
        encoded_text: str = cls.b85_encode(compressed_bytes)
        return encoded_text

    def readable_value(self, data: str) -> str:
        """Decompress the encoded string and reverse the decoded string"""
        decoded_bytes = self.b85_decode(data)
        decompressed_bytes = self.z_decompress(decoded_bytes)
        deserialized_bytes = msgpack.unpackb(decompressed_bytes)
        decoded_text = deserialized_bytes.decode("utf-8")
        return decoded_text

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
