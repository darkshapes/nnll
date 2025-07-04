# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from nnll.metadata.helpers import make_callable
from nnll.tensor_pipe.parenting import seek_class_path


def test_seek_diffusers_path():
    assert seek_class_path(make_callable("AllegroPipeline", "diffusers"), "diffusers") == ["diffusers", "pipelines", "allegro"]


def test_seek_transformers_path():
    assert seek_class_path(make_callable("AlbertModel", "transformers"), "transformers") == ["transformers", "models", "albert"]


def test_seek_class_attention():
    assert seek_class_path("CogVideoXAttnProcessor2_0", "diffusers") is None
