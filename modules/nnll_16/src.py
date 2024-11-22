
from abc import ABC, abstractmethod
import torch

class Backend(ABC):
    """Dynamically retrieve and set attributes from torch."""
    def __init__(self, backend_type: str):
        """
        #### Class structured around iterating methods of Pytorch `torch` function
        #### `backend_type`: device suffix received from subclasses
        #### OUTPUT:
        """
        self._is_available = False
        self._is_built = False
        self._device_count = 0
        self._get_device_name = None
        self.backend_type = backend_type
        self.torch_exists = ["is_available"]
        self.torch_built = [f"backends.{backend_type}.is_available"]
        self.torch_count = ["device_count"]
        self.configure()

    @abstractmethod
    def configure(self):
        pass

    def attribute(self, m: str) -> any:
        """
        #### Dynamically set class variables based on system configuration
        #### `m`: A valid PyTorch method as specified in `torch_methods` subclasses
        #### OUTPUT: Boolean values assigned to respective main class variables
        """
        try:
            method = getattr(torch, f"{self.backend_type}.{m}")
            result = method()
            setattr(self, f"_{m}", result)
            return result
        except AttributeError as e:
            print(f"AttributeError: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

class CUDADevice(Backend):
    def __init__(self): super().__init__("cuda")

    def configure(self):
        if self.attribute(self.torch_exists) and self.attribute(self.torch_built):
            torch_methods = [f"get_device_properties({self.backend_type}).total_memory", "get_device_name"]
            device_count = self.attribute(self.torch_count)
            for _ in range(device_count):
                for m in torch_methods:
                    self.attribute(m)
            super().__init__("backends.cuda")
            torch_methods = ["is_flash_attention_available", "mem_efficient_sdp_enabled"]
            for m in torch_methods:
                self.attribute(m)

class MPSDevice(Backend):
    def __init__(self): super().__init__("mps")

    def configure(self):
        if self.attribute(self.torch_exists) and self.attribute(self.torch_built):
            torch_methods = ["recommended_max_memory"]
            for m in torch_methods:
                self.attribute(m)
            super().__init__("backends.mps")
            torch_methods = ["enable_attention_slicing"]
            for m in torch_methods:
                self.attribute(m)

class XPUDevice(Backend):
    def __init__(self): super().__init__("xpu")

    def configure(self):
        if self.attribute(self.torch_exists):
                torch_methods = ["max_memory_reserved", "get_device_name"]
                device_count = self.attribute(self.torch_count)
                for _ in range(device_count):
                    for m in torch_methods:
                        self.attribute(m)


# directml
# openvino

cuda_device = CUDADevice()
print(f"Is available: {cuda_device._is_available}")
print(f"Is built: {cuda_device._is_built}")
print(f"Device count: {cuda_device._device_count}")
print(f"Get device name: {cuda_device._get_device_name}")



# class Backend(ABC):
#     def __init__(self, backend_type: str):
#         self._is_available = False
#         self._is_built = False
#         self._device_count = 0
#         self._get_device_name = None
#         self.backend_type = backend_type
#         self.configure()

#         self.attribute = lambda m: getattr(torch, f"{self.backend_type}.{m}")()

#     @abstractmethod # this makes configure required in subclasses
#     def configure(self):
#         pass

# class CUDADevice(Backend):
#     def __init__(self): super().__init__("cuda")

#     def configure(self):
#         torch_methods = ["is_available", "is_built", "device_count", "get_device_name"]
#         for m in torch_methods:
#             self.attribute(m)

# class XPSDevice(Backend):
#     def __init__(self): super().__init__("xps")

#     def configure(self):
#         try:
#             self._is_available = getattr(torch, f"{self.backend_type}.is_available")()
#             self._is_built = True
#             self._device_count = getattr(torch, f"{self.backend_type}.device_count")()
#         except ImportError:
#             print(f"Module {self.backend_type} is not installed.")
#             self._is_available = False

# class MPSDevice(Backend):
#     def __init__(self): super().__init__("cuda")

#     def configure(self):
#         self._is_available = getattr(torch, f"{self.backend_type}.is_available")()
#         self._is_built = True
#         self._device_count = getattr(torch, f"{self.backend_type}.device_count")()

# # Example usage
# backend_cuda = CudaBackend()
# print("CUDA Is Available:", backend_cuda.is_available())  # This will print True if CUDA is available
# backend_xps = XpsBackend()
# print("XPS Device Count:", backend_xps.get_device_count())  # This will print 0 if XPS module is not installed


