#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import pytest
from nnll.tensor_pipe.deconstructors import root_class


def test_root_class_with_builtin_types():
    class DummyInitModule:
        def __init__(self, flag: bool, count: int):
            pass

    expected_output = {}

    result = root_class(DummyInitModule)
    assert result == expected_output


if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
