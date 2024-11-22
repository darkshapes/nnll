
from nnll_16.src import Backend

class CUDADevice(Backend):
    def __init__(self): super().__init__("cuda")

    def configure(self):
        self.attribute()
        if self.attribute("is_available") == True and self.attribute("is_built", True) == True:
            torch_methods = [f"get_device_properties({self.backend_type}).total_memory", "get_device_name"]
            device_count = self.attribute(self.torch_count)
            for i in range(device_count):
                for m in torch_methods:
                    self.attribute(m)
            torch_methods = ["is_flash_attention_available", "mem_efficient_sdp_enabled"]
            for m in torch_methods:

                self.attribute(m)
class MPSDevice(Backend):
    def __init__(self): super().__init__("cuda")

    def configure(self):
        if self.torch_exists() == True and self.torch_built() == True:
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

class DMLDevice(Backend):
    def __init__(self): super().__init__("dml")

    def configure(self):
        if self.attribute(self.torch_exists):
            elif hasattr(torch, "dml") and torch.dml.is_available():  # type: ignore

class OPENVINODevice(Backend):
    import openvino as ov
    from modules.intel.openvino import get_device as get_raw_openvino_device


cuda_device = MPSDevice()
print(f"{cuda_device}")



"""
class contains torch value and getattr
we need to pass cuda and backends on demand to getattr
"""
# self._is_available = False
# self._is_built = False
# self._device_count = 0
# self._get_device_name = None
# self._is_flash_attention_available = False
# self._mem_efficient_sdp_enabled = False
# self._enable_attention_slicing = False
# self._max_recommended_memory = 0
# self._max_memory_reserved = 0