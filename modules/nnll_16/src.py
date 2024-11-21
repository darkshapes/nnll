
from abc import ABC, abstractmethod
import torch

class Backend(ABC):
    def __init__(self, backend_type: str):
        self._is_available = False
        self._is_built = False
        self._device_count = 0
        self.backend_type = backend_type
        self.configure()
        self.is_available = lambda: getattr(torch, f'{self.backend_type}.is_available')()
        self.is_built = lambda: getattr(torch, f'{self.backend_type}.is_built')()
        self.get_device_count = lambda: getattr(torch, f'{self.backend_type}.device_count')()

class CUDABackend(Backend):
    def __init__(self): super().__init__('cuda')

    def configure(self):
        self._is_available = self.is_available
        self._is_built = self.is_built
        self.get_device_count = self.get_device_count
        self.get_device_name = self.get_device_name

class XPSBackend(Backend):
    def __init__(self): super().__init__('xps')

    def configure(self):
        try:
            self._is_available = getattr(torch, f'{self.backend_type}.is_available')()
            self._is_built = True
            self._device_count = getattr(torch, f'{self.backend_type}.device_count')()
        except ImportError:
            print(f"Module {self.backend_type} is not installed.")
            self._is_available = False

class MPSBackend(Backend):
    def __init__(self): super().__init__('cuda')

    def configure(self):
        self._is_available = getattr(torch, f'{self.backend_type}.is_available')()
        self._is_built = True
        self._device_count = getattr(torch, f'{self.backend_type}.device_count')()

# Example usage
backend_cuda = CudaBackend()
print("CUDA Is Available:", backend_cuda.is_available())  # This will print True if CUDA is available
backend_xps = XpsBackend()
print("XPS Device Count:", backend_xps.get_device_count())  # This will print 0 if XPS module is not installed



torch.backends.mps.enable_attention_slicing
torch.backends.cuda.flash_sdp_enabled() #flash
torch.backends.cuda.mem_efficient_sdp_enabled() #xformers
torch.cuda.mem_get_info()[1] #cuda mem

torch.mps.driver_allocated_memory() #mps
torch.mps.recommended_max_memory()

torch.version.hip # (for rocm)

torch.cuda.get_device_name(torch.device("cuda")).endswith("[ZLUDA]")

# directml
# openvino
