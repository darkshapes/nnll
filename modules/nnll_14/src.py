
import sys
import random
import torch

def supported_backends():
    """
    #### Initial check of system hardware. Counts all gpus and cpus if available.
    #### Precursor to further ZLUDA/XPS/ROCM|HIP discernment
    #### OUTPUT: iterates string values of available devices
    """
    possible = ["cuda", "cpu", "mps"] # The most common processing frameworks
    print(sys.platform)
    compatible = possible[:-1] if sys.platform.lower() != "darwin" else possible[1:] # Skip impossible gpu combinations
    for backend in compatible:
        if not hasattr(torch, backend):
            raise ValueError(f"Unsupported or unavailable backend: {backend}")
        else:
            backend_module = getattr(torch, backend)
            if (hasattr(backend_module, "is_available") and not backend_module.is_available()): # Is the device present?
                raise RuntimeError(f"{backend.lower()} is not an available device.")
                continue
            if (hasattr(backend_module, "is_built") and not backend_module.is_built()): # Are drivers working?
                raise RuntimeError(f"{backend.lower()} is an available but not a configured device.")
            else:
                if hasattr(backend_module, "device_count()"):
                    for i in range(backend_module.device_count()):
                        yield f"{backend}:{i}"
                else:
                    yield backend