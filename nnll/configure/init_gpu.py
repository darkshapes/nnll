# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


from typing import Callable, Union

from nnll.monitor.console import nfo
from nnll.monitor.file import debug_monitor


@debug_monitor
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


@debug_monitor
def random_int_from_gpu(input_seed: int = soft_random()) -> int:
    """
    Generate a random number via pytorch
    :params input_seed: `int`a seed to feed the random generator, defaults to `soft_random()` function
    :returns: `int` A random number from current device
    """
    import torch

    torch.set_num_threads(1)

    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)


@debug_monitor
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
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if "mps" in device:
        torch.mps.manual_seed(seed)
        torch.backends.mps.torch.use_deterministic_algorithms(deterministic)
    if "xpo" in device:
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

    return seed


@debug_monitor
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


@debug_monitor
def first_available(processor: str = None, assign: bool = True, clean: bool = False, init: bool = True) -> Union[Callable, str]:
    """Return first available processor of the highest capacity\n
    :param processor: Name of an existing processing device, defaults to None (autodetect)
    :param assign: Direct torch to use the detected device, defaults to True
    :param clean: Clear any previous cache, defaults to False
    :param init: Initialize the device with a test tensor and discard, defaults to True\n
    :return: The torch device handler, or the name of the processor

    """
    from functools import reduce

    import torch

    torch.set_num_threads(1)
    if not processor:
        processor = reduce(
            lambda acc, check: check() if acc == "cpu" else acc,
            [
                lambda: "cuda" if torch.cuda.is_available() else "cpu",
                lambda: "mps" if torch.backends.mps.is_available() else "cpu",
                lambda: "xpu" if torch.xpu.is_available() else "cpu",
                lambda: "mtia" if torch.mtia.is_available() else "cpu",
            ],
            "cpu",
        )

    if clean:
        import gc

        gc.collect()
        if processor == "cuda":
            torch.cuda.empty_cache()
        if processor == "mps":
            torch.mps.empty_cache()
        if processor == "xpu":
            torch.xpu.empty_cache()
        if processor == "mtia":
            torch.mtia.empty_cache()

    if init:
        tensor = random_tensor_from_gpu(device=processor)
        if tensor:
            del tensor
        tensor = None

    nfo(f"Available torch devices: {processor}")
    return torch.device(processor) if assign else processor
