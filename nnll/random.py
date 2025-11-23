# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Random number generation utilities with centralized control."""

from typing import Literal


class RNGState:
    """Centralized random number generator controller and state management.\n"""

    def __init__(self, device: str = "cpu", initial_seed: int | None = None, deterministic: bool = False, source: Literal["torch", "mps"] = "torch"):
        """Initialize the random generator.\n
        :param device: Device to use for torch operations ("cpu", "cuda", or "mps")
        :param initial_seed: Optional initial seed. If None, generates a random seed.
        """
        import torch

        self._seed: int | None = initial_seed
        self.set_device(device)
        self._source = source
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.mps.torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic

    @property
    def seed(self) -> int | None:
        """Reveal the current seed value\n
        :returns: Current seed, or None if not set
        """
        return self._seed

    def next_seed(self, seed: int | None = None) -> int:
        """Set seed across all random generators and devices.\n
        :param seed: Seed value. If None, generates a random seed using numpy_random.
        :param deterministic: Whether to use deterministic algorithms (currently unused, reserved for future use)
        :returns: The seed value that was set
        """
        import torch

        if seed is None:
            seed = self._torch_generator.seed()

        self._seed = seed

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)
        if "mps" in self.device and torch.backends.mps.is_available():
            torch.mps.manual_seed(self._seed)

        return self._seed

    def set_device(self, device: str) -> None:
        """Set the device for torch operations.\n
        :param device: Device string ("cpu", "cuda", or "mps")
        """
        import torch
        from torch import Generator

        self.device = device
        self._torch_generator = None  # Reset generator for new device
        self._torch_generator: Generator | None = torch.Generator(device=self.device)
