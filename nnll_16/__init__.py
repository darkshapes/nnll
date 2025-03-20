#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


def first_available(processor=None):
    from functools import reduce
    import torch

    if not processor:
        processor = reduce(
            lambda acc, check: check() if acc == "cpu" else acc, [lambda: "cuda" if torch.cuda.is_available() else "cpu", lambda: "mps" if torch.backends.mps.is_available() else "cpu", lambda: "xpu" if torch.xpu.is_available() else "cpu"], "cpu"
        )
    return torch.device(processor)
