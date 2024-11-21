
"""
soft_random
Returns a deterministic random number generated using philox method
`size` : RNG ceiling

hard_random
Returns a non-prng random number generated using cryptographic secrets
`hardness` : byte length of generated number

gpu_random
Return a random number generated via torch

tensor_random
Create a random tensor ( not sure why )
`input_seed` the seed to control randomization

seed_planter
Ensure seed gets planted in every gen library
`input_seed` the seed to plant
`deterministic` toggle the algorithm to force deterministic or not
"""

import secrets as secrets
from functools import cache as function_cache, wraps
import random
from numpy.random import SeedSequence, Generator, Philox

def soft_random(size=0x2540BE3FF):
    entropy = f"0x{secrets.randbits(128):x}"  # good entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))

def hard_random(hardness=5):
    return int(secrets.token_hex(hardness), 16)  # make hex secret be int

def gpu_random(input_seed=None):
    from torch import torch
    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)

def random_tensor(device, input_seed=None):
     x = random.seed().randrange(-4,4, 1)
     y = random.seed().randrange(-4,4, 1)
     return torch.rand(size=(y, y)).to(device)


def seed_planter(input_seed:int=None, deterministic:bool=True):
    if input_seed is None: input_seed = soft_random()
    random.seed(input_seed)
    from torch import torch
    torch.manual_seed(input_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        return torch.cuda.manual_seed(input_seed), torch.cuda.manual_seed_all(input_seed)
    elif torch.backends.mps.is_available():
        return torch.mps.manual_seed(input_seed)
    elif torch.xpu.is_available():
         return torch.xpu.manual_seed(input_seed)



