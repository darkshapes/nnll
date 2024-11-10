import random
import numpy

import secrets as secrets

from numpy.random import SeedSequence, Generator, Philox

def soft_random(size=0x2540BE3FF): # returns a deterministic random number using Philox
    entropy = f"0x{secrets.randbits(128):x}" # git gud entropy
    rndmc   = Generator(Philox(SeedSequence(int(entropy,16))))
    return int(rndmc.integers(0, size))

def hard_random(hardness=5): # returns a non-prng random number use secrets
    return int(secrets.token_hex(hardness),16) # make hex secret be int

def tensor_random(seed=None):
    from torch import torch
    return torch.random.seed() if seed is None else torch.random.manual_seed(seed)

def tensorify(hard, size=4): # Creates an array of default size 4x1 using either softRandom or hardRandom
    num = []
    for s in range(size): # make array, convert float, negate it randomly
        if hard==False: # divide 10^10, truncate float
            conv = '{0:.6f}'.format((float(soft_random()))/0x2540BE400)
        else:  # divide 10^12, truncate float
            conv = '{0:.6f}'.format((float(hard_random()))/0xE8D4A51000)
        num.append(float(conv)) if secrets.choice([True, False]) else num.append(float(conv)*-1)
    return num

def seed_planter(seed, deterministic=True):
    from torch import torch
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available()==True:
        if deterministic == True:
            return {'torch.backends.cudnn.deterministic': 'True','torch.backends.cudnn.benchmark': 'False'}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available()==True:
        return torch.mps.manual_seed(seed)
    # elif torch.xpu.is_available():
    #     return torch.xpu.manual_seed(seed)


### SERVER INFORMATION ROUTINES ###

### TODO: this stuff should all go in config.py or related somewhere, maybe device.py?