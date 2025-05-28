### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


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


@debug_monitor
def soft_random(size: int = 0x100000000) -> int:  # previously 0x2540BE3FF
    """
    Generate a deterministic random number using philox\n
    :params size: `int` RNG ceiling in hex format
    :returns: `int` a random number of the specified length\n
    pair with `random.seed()` for best effect
    """

    import secrets
    from numpy.random import SeedSequence, Generator, Philox

    entropy = f"0x{secrets.randbits(128):x}"  # good entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))


@debug_monitor
def seed_planter(seed, deterministic: bool = True) -> int:
    from numpy import random
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

    return seed


@debug_monitor
def hard_random(hardness: int = 5) -> int:
    """
    Generate a cryptographically secure random number\n
    :param hardness: `int` byte length of generated number
    :returns: `int` Non-prng random number
    """
    import secrets

    return int(secrets.token_hex(hardness), 16)  # make hex secret be int


# @debug_monitor
# def c(dtype: str) -> torch.dtype:
#     return {}.get(dtype)
