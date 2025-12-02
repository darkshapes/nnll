# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Test reverse codec"""

# pylint:disable=line-too-long, import-outside-toplevel

from nnll.reverse_codec import ReversibleBytes


def test_reverse_codec_encodes():
    text = "Hello, World!"
    reversible_bytes = ReversibleBytes(text)
    assert isinstance(reversible_bytes, ReversibleBytes)
    assert reversible_bytes.value == b"5YN(V30KcK1%3g%3Qv8@B&{V1%c"


def test_reverse_codec_reverses():
    text = "Hello, World!"
    reversible_bytes = ReversibleBytes(text)
    assert reversible_bytes.value == b"5YN(V30KcK1%3g%3Qv8@B&{V1%c"
    byte_data = reversible_bytes.value
    assert reversible_bytes.decompress_state(byte_data) == text
