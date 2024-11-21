
import torch
import sys

class Backend:
    def is_available(self):
        return False

    def is_built(self):
        return False

check_function = lambda module, attr_name: (hasattr(module, attr_name) and not getattr(module, attr_name)())

def check_attribute(module, attr_name, error_out, error_msg):
    if hasattr(module, attr_name) and callable(getattr(module, attr_name)):
        return getattr(module, attr_name)()
    raise error_out(error_msg); return False

is_built_result = check_attribute(backend_module, "is_built")
is_available_result = check_attribute(backend_module, "is_available")

def supported_backends():
    """
    #### Initial check of system hardware. Counts all gpus and cpus if available.
    #### Precursor to further ZLUDA/XPS/ROCM|HIP discernment
    #### OUTPUT: iterates string values of available devices
    """
    possible = {
                "is_available": ["xps", "cuda", "cpu", "mps" ],
                "is_built" : ["cuda", "mps"],
                "device_count": ["xps", "cuda", "cpu", "mps" ]
                "mps": "is_available", "is_built", "device_count", "recommended_max_memory"
                }

    # Processing frameworks

    psutil.virtual_memory().total # cpu mem
    compatible = possible[:-1] if sys.platform.lower() != "darwin" else possible[1:] # Skip impossible gpu combinations
    for backend in compatible:
            backend_module = check_attribute(torch, backend, ValueError(f"Unsupported or unavailable backend: {backend}"))
            if backend_module != False:
                check_attribute(backend_module, "is_available", RuntimeError, (f"{backend.lower()} is not an available device.")) # Is the device present?
                check_attribute(backend_module, "is_built", RuntimeError, (f"{backend.lower()} is an available but not a configured device.")) # Is the device setup?
                device_count = check_attribute(backend_module, "device_count", RuntimeError, (f"{backend.lower()} cant't count devices on this framework")) # Are there more than one?
                if device_count != False:
                    for i in range(device_count):
                        device_name = check_attribute(backend_module, "get_device_name", RuntimeError, (f"{backend.lower()} names unavailale on this device framework."))
                        yield device_name if device_name !=False else f"{backend}:{i}"
                    else:
                        yield backend

