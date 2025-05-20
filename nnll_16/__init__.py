#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable
from nnll_01 import debug_monitor, nfo


@debug_monitor
def first_available(processor: str = None) -> Callable:
    """Return first available\n

    :param processor: _description_, defaults to None
    :return: _description_
    """
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
    nfo(f"highest available torch device: {processor}")
    if processor == "mps":
        torch.mps.set_per_process_memory_fraction(1.7)
    return torch.device(processor)


# @debug_monitor
# def c(dtype: str) -> torch.dtype:
#     return {}.get(dtype)
