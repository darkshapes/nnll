### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from nnll_01 import debug_monitor


@debug_monitor
def random_int_from_gpu(input_seed: int = None) -> int:
    """
    Generate a random number via pytorch
    :params input_seed: `int`a seed to feed the random generator
    :returns: `int` a random number generated
    """
    import torch

    return torch.random.seed() if input_seed is None else torch.random.manual_seed(input_seed)


@debug_monitor
def random_tensor_from_gpu(device: str = "cpu", input_seed: int = None):
    """
    Create a random tensor shape (for testing or other purposes)\n
    :params input_seed: `int` the seed to control randomization
    :params device: `str` device to assign generation to
    :returns: `tensor` random dimensional tensor
    """

    import torch

    if input_seed is not None:
        torch.manual_seed(input_seed)
    return torch.rand(1, device=device)
