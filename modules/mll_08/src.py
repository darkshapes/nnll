
import secrets as secrets
from functools import cache as function_cache, wraps
import random
from numpy.random import SeedSequence, Generator, Philox

"""
soft_random
Returns a deterministic random number using Philox
`size` : RNG ceiling

hard_random
Returns a non-prng random number using cryptographic secrets
`hardness` : byte length of generated number

tensor_random
Create a randomly shaped tensor tensor

seed_planter
Ensure seed gets planted in every gen library
"""


def soft_random(size=0x2540BE3FF):
    entropy = f"0x{secrets.randbits(128):x}"  # git gud entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))


def hard_random(hardness=5):
    return int(secrets.token_hex(hardness), 16)  # make hex secret be int


def tensor_random(seed=None):
    from torch import torch
    return torch.random.seed() if seed is None else torch.random.manual_seed(seed)


def seed_planter(seed, deterministic=True):
    from torch import torch
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() == True:
        if deterministic == True:
            return {'torch.backends.cudnn.deterministic': 'True', 'torch.backends.cudnn.benchmark': 'False'}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available() == True:
        return torch.mps.manual_seed(seed)
    # elif torch.xpu.is_available():
    #     return torch.xpu.manual_seed(seed)
