#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import pytest
from typing import List
from nnll.metadata.helpers import class_parent  # Replace with the actual module name


def test_class_parent_diffusers():
    assert class_parent("stable-diffusion", "Diffusers") == ["diffusers", "pipelines", "stable_diffusion"]


def test_class_parent_transformers():
    assert class_parent("albert", "Transformers") == ["transformers", "models", "albert"]


def test_class_parent_invalid_parent():
    with pytest.raises(KeyError):
        class_parent("unknown", "Unknown")


def test_class_parent_empty_parent():
    with pytest.raises(KeyError):
        assert class_parent("", "") == ["", "", ""]


def test_class_parent_bad_code_name():
    assert class_parent("diffdusers", "diffusers") is None


def test_class_parent_mixed_case():
    assert class_parent("sana", "DIFFusERS") == ["diffusers", "pipelines", "sana"]


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
