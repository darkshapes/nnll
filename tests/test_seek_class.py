import pytest
from enum import Enum
from unittest.mock import patch
from mir.mappers import make_callable
from nnll.tensor_pipe.deconstructors import seek_class_path


def test_seek_diffusers_path():
    assert seek_class_path(make_callable("AllegroPipeline", "diffusers"), "diffusers") == ["diffusers", "pipelines", "allegro"]


def test_seek_transformers_path():
    assert seek_class_path(make_callable("AlbertModel", "transformers"), "transformers") == ["transformers", "models", "albert"]


def test_seek_class_attention():
    assert seek_class_path("CogVideoXAttnProcessor2_0", "diffusers") is None
