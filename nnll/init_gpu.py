# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import torch
import gc
from typing import Literal


def set_torch_device(
    device_override: Literal["cuda", "mps", "cpu"] | None = None,
) -> torch.device:
    """Set the PyTorch device, with optional manual override.\n
    :param device_override: Optional device to use. "cuda", "mps", or "cpu"
    :returns: The selected torchdevice
    :raises ValueError: If device_override is not one of the allowed values"""
    if device_override is not None:
        if device_override not in ("cuda", "mps", "cpu"):
            raise ValueError(f"device_override must be one of 'cuda', 'mps', or 'cpu', got '{device_override}'")
        return torch.device(device_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    return device


device = set_torch_device()
dtype = torch.float16


def sync_torch(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cpu":
        pass


def clear_cache(device_override: Literal["cuda", "mps", "cpu"] | None = None):
    if device.type == "cuda" or device_override == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" or device_override == "mps":
        torch.mps.empty_cache()
    gc.collect()
