# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from __future__ import annotations

from typing import Literal

import torch


class Gfx:
    """
    Detects the best available Torch device (CUDA → MPS → CPU) at instantiation time and provides a matching default dtype.\n
    `>>> gfx = Gfx()`
    `>>> gfx.device`
    device(type='cuda')
    `>>> gfx.dtype`
    torch.float16
    """

    _PREFERENCE: tuple[tuple[bool, Literal["cuda"]], tuple[bool, Literal["mps"]], tuple[bool, Literal["cpu"]]] = (
        (torch.cuda.is_available(), "cuda"),
        (torch.backends.mps.is_available(), "mps"),
        (True, "cpu"),  # always fallback to CPU
    )

    def __init__(self, full_precision: bool = False, device: str | None = None) -> None:
        """Resolve the device once; store as a private attribute
        :param full_precision: whether to use full precision (float32) instead"""

        self.full_precision = full_precision
        if device is not None:
            self._device = torch.device(device)
        for available, name in self._PREFERENCE:
            if available:
                self._device: torch.device = torch.device(name)
                break

    @property
    def auto(self) -> torch.device:
        """Return the selected torch device (e.g. ``torch.device('cuda')``)."""
        return self._device

    @property
    def dtype(self, full_precision: bool = False) -> torch.dtype:
        """Retrieve a compatible dtype for the device
        :param full_precision: whether to use full precision (float32) instead
        :returns: A sensible default dtype for the selected device
        """
        if self._device.type == "cuda":
            return torch.float16 if not self.full_precision else torch.float32
        if self._device.type == "mps":
            return torch.bfloat16 if not self.full_precision else torch.float32
        return torch.float32

    @property
    def sync(self) -> None:
        """Synchronise the current device."""
        device_type = getattr(torch, self._device.type)
        device_type.synchronize()

    @property
    def empty_cache(self) -> None:
        """Clear the device cache and run garbage collection."""
        import gc

        device_type = getattr(torch, self._device.type)
        device_type.synchronize()
        gc.collect()

    @property
    def device(self) -> torch.device:
        """Get the current torch device (e.g. ``torch.device``)"""
        return self._device
