
import torch
import random


def gpu_random(input_seed: int = None) -> int:
    """
    Generate a random number via pytorch
    :params input_seed: `int`a seed to feed the random generator
    :returns: `int` a random number generated
    """
    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)


def random_tensor(device: str = "cpu", input_seed: int = None) -> torch.Tensor:
    """
    Create a random tensor shape (for testing or other purposes)\n
    :params input_seed: `int` the seed to control randomization
    :params device: `str` device to assign generation to
    :returns: `tensor` random dimensional tensor
    """
    if input_seed is not None:
        torch.manual_seed(input_seed)
    return torch.rand(1, device=device)
