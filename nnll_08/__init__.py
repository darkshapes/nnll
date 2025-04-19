### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from nnll_01 import debug_monitor


@debug_monitor
def soft_random(size: int = 0x2540BE3FF) -> int:
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
def seed_planter(seed, deterministic=True) -> int:
    from numpy import random
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        if deterministic:
            return {"torch.backends.cudnn.deterministic": "True", "torch.backends.cudnn.benchmark": "False"}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        return torch.mps.manual_seed(seed)
    elif torch.xpu.is_available():
        return torch.xpu.manual_seed(seed)


@debug_monitor
def hard_random(hardness: int = 5) -> int:
    """
    Generate a cryptographically secure random number\n
    :param hardness: `int` byte length of generated number
    :returns: `int` Non-prng random number
    """
    import secrets

    return int(secrets.token_hex(hardness), 16)  # make hex secret be int
