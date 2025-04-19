#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from nnll_01 import debug_monitor


@debug_monitor
def first_available(processor=None):
    from functools import reduce
    import torch

    if not processor:
        processor = reduce(
            lambda acc, check: check() if acc == "cpu" else acc,
            [
                lambda: "cuda" if torch.cuda.is_available() else "cpu",
                lambda: "mps" if torch.backends.mps.is_available() else "cpu",
                lambda: "xpu" if torch.xpu.is_available() else "cpu",
            ],
            "cpu",
        )
    return torch.device(processor)


# @debug_monitor
# def c(dtype: str) -> torch.dtype:
#     return {}.get(dtype)
