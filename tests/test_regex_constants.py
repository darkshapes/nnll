# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from nnll.configure.constants import PARAMETERS_SUFFIX
from nnll.mir.tag import make_mir_tag


def test_constants():
    import re

    data_tests = {
        "mlx-community/Kokoro-82M-4bit": ["kokoro", "*"],
        "RuadaptQwen2.5-32B-Pro-Beta:latest": ["ruadaptqwen2", "*"],
        "microsoft/Phi-4-mini-instruct": ["phi-4", "*"],
        "tiiuae/falcon-mamba-7b": ["falcon-mamba", "*"],
        "ijepa-vith14-1k": ["ijepa-vith14", "*"],
        "arcee-ai/AFM-4.5B": ["afm", "*"],
        "ibm-research/PowerMoE-3b": ["powermoe", "*"],
        "qwen1-5-moe-a2-7b": ["qwen1-5-moe-a2", "*"],
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": ["sana-sprint-1024px", "diffusers"],
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers": ["hunyuandit-v1", "diffusers"],
    }
    # regex = PARAMETERS_SUFFIX
    for test, expected in data_tests.items():
        mir_tag = list(make_mir_tag(test))
        assert mir_tag == expected
