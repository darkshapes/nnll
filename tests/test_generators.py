# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def test_list_diffusers_models():
    __import__("nnll").mir.indexers.diffusers_index()
    __import__("nnll").mir.indexers.transformers_index()
