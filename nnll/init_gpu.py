# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Literal, Optional, Any


def soft_random(size: int = 0x100000000) -> int:  # previously 0x2540BE3FF
    """
    Generate a deterministic random number using philox\n
    :params size: `int` RNG ceiling in hex format
    :returns: `int` a random number of the specified length\n
    pair with `random.seed()` for best effect
    """

    import secrets

    from numpy.random import Generator, Philox, SeedSequence

    entropy = f"0x{secrets.randbits(128):x}"  # good entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))


def hard_random(hardness: int = 5) -> int:
    """
    Generate a cryptographically secure random number\n
    :param hardness: `int` byte length of generated number
    :returns: `int` Non-prng random number
    """
    import secrets

    return int(secrets.token_hex(hardness), 16)  # make hex secret be int


def random_int_from_gpu(input_seed: int = soft_random()) -> int:
    """
    Generate a random number via pytorch
    :params input_seed: `int`a seed to feed the random generator, defaults to `soft_random()` function
    :returns: `int` A random number from current device
    """
    import torch

    torch.set_num_threads(1)

    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)


def seed_planter(seed: int = soft_random(), deterministic: bool = False, device: str = "cpu") -> int:
    """Force seed number to all available devices\n
    :param seed: The number to grow all random generation from, defaults to `soft_random` function
    :param deterministic: Identical number provides identical output, defaults to True
    :param device: Processor to use, defaults to `first_available(assign=False)` function
    :return: The `int` seed that was provided to the functions.
    """
    import torch
    from numpy import random

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if "mps" in device:
        torch.mps.manual_seed(seed)

    return seed


def random_tensor_from_gpu(device: str, input_seed: int = soft_random()):
    """
    Create a random tensor shape (for testing or other purposes)\n
    :params device: `str` device to assign generation to, defaults to `first_available()` function
    :params input_seed: `int` the seed to control randomization, defaults to soft_random() generator
    :returns: `tensor` Random dimensional tensor
    """

    import torch

    torch.set_num_threads(1)
    if input_seed is not None:
        seed_planter(device=device)
    return torch.rand(1, device=device)


def set_torch_device(
    device_override: Optional[Literal["cuda", "mps", "cpu"]] = None,
) -> Any:
    """Set the PyTorch device, with optional manual override.\n
    :param device_override: Optional device to use. "cuda", "mps", or "cpu"
    :returns: The selected torchdevice
    :raises ValueError: If device_override is not one of the allowed values"""
    import torch

    if device_override is not None:
        return torch.device(device_override)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


device = set_torch_device()


def clear_cache(device_override: Optional[Literal["cuda", "mps", "cpu"]] = None):
    import gc
    import torch

    gc.collect()
    if device.type == "cuda" or device_override == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" or device_override == "mps":
        torch.mps.empty_cache()
