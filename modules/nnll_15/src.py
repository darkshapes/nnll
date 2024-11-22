
import torch
import sys
import psutil

def supported_backends():
    """
    #### Initial check of system hardware. Counts all gpus and cpus if available.
    #### Precursor to further ZLUDA/XPS/ROCM|HIP discernment if needed
    #### OUTPUT: iterates string values of available devices
    """

    torch_pc = {
                "is_available": ["xpu", "cuda"]
    }

    torch_darwin = {
                "is_available": ["mps"]
    }

    torch_modules = {
                "device_count": ["xpu", "cuda", "mps" ],
                "recommended_max_memory" : ["mps"]
                }

    torch_backend_modules = {
                "is_built" : ["cuda", "mps"],
    }

    # check_function = lambda module, attr_name: (hasattr(module, attr_name) and not getattr(module, attr_name)())

    def check_attribute(module, attr_name, error_out, error_msg):
        if hasattr(module, attr_name) and callable(getattr(module, attr_name)):
            return getattr(module, attr_name)()
        raise error_out(error_msg); return False

    # Processing frameworks
    psutil.virtual_memory().total # CPU mem, safe to assume there is one CPU, generally only one found on consumer hardware
    compatible = torch_pc if sys.platform.lower() != "darwin" else torch_darwin # Skip impossible gpu combinations
    for backend in compatible:
            torch_module = check_attribute(torch, backend, ValueError(f"Unsupported or unavailable backend: {backend}"))
            backend_module = check_attribute(torch.backends, backend, ValueError(f"Unsupported or unavailable backend: {backend}"))

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


support= supported_backends()
