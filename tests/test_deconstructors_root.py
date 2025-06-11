from typing import Annotated
import pytest
from typing import Generic
from nnll.tensor_pipe.deconstructors import root_class


def test_root_class_with_builtin_types():
    class DummyInitModule:
        def __init__(self, flag: bool, count: int):
            pass

    expected_output = {}

    result = root_class(DummyInitModule)
    assert result == expected_output


# def test_root_class_with_complex_types():
#     Annotated[T, "model"]

#     class TestClass:
#         Optimized[T] = Annotated[T, runtime.Optimize()]
#         pass

#     # type checker will treat Optimized[int]
#     # as equivalent to Annotated[int, runtime.Optimize()]

#     expected_output = {"model": ["😹"], "tokenizer": ["str"], "config": ["dict"]}

#     result = root_class(TestClass)
#     assert result == expected_output
