# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Reverse codec for compressing and decompressing data"""

# pylint:disable=line-too-long, import-outside-toplevel

from dataclasses import dataclass
import zlib
import base64


@dataclass
class ReversibleBytes:
    value: bytes

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
        return base64.b85encode(zlib.compress(text.encode()))[::-1]

    def decompress_state(self, data: bytes) -> str:
        """Decompress the encoded string and reverse the decoded string\n
        :param encoded_text: The encoded text to decompress
        :return: Decompressed translated text"""
        reversed_value = data[::-1]
        return zlib.decompress(base64.b85decode(reversed_value)).decode()
