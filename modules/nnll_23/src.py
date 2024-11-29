
from functools import reduce
from typing import Callable
import sys
import importlib


class DynamicMethodConstructor:
    def __init__(self):
        self.methods = {}

    def load_method(self, method_name: str, module_path: str, attribute_path: str) -> None:
        """
        Load a method dynamically based on the module path and attribute path.\n
        :param method_name: Name that will refer to the loaded method
        :param module_path: Path to the module (e.g., 'os.path')
        :param attribute_path: Dot-separated string representing the path to the method within the module (e.g., 'join')
        :return: None, enlivens the method_name with the module and attributes requested

        """
        # Check if the module is already in sys.modules
        if module_path in sys.modules:
            print("module already loaded, skipping...")
        try:
            # Import the module and get the attribute
            module = importlib.import_module(module_path)
            method = reduce(getattr, attribute_path.split('.'), module)
            self.methods[method_name] = method

        except ImportError as error_log:
            raise RuntimeError(f"Failed to load module '{module_path}': {error_log}")

        except AttributeError as error_log:
            raise RuntimeError(f"Failed to access attribute '{attribute_path}' in module '{module_path}': {error_log}")

    def call_method(self, method_name: str, *args, **kwargs) -> Callable:
        """
        Call a dynamic construction from `load_method` with provided arguments.\n
        :param method_name: Name of the method to be called
        :param args: Positional arguments for the method
        :param kwargs: Keyword arguments for the method
        :return: Result of the method call
        """
        if method_name in self.methods:
            return self.methods[method_name](*args, **kwargs)
        else:
            raise AttributeError(f"Method '{method_name}' not found.")


# Example usage
if __name__ == "__main__":
    constructor = DynamicMethodConstructor()
    # Load methods dynamically based on system specifications or available files
    d = constructor.load_method('cuda_available', 'torch.cuda', 'is_available')
    constructor.load_method('cuda_exists', 'torch.backends.cuda', 'is_built')
    constructor.load_method('mps_available', 'torch.mps', 'is_available')
    constructor.load_method('mps_exists', 'torch.backends.mps', 'is_built')
    print(constructor.call_method('cuda_available'))
    print(constructor.call_method('mps_available'))
    construct_two = DynamicMethodConstructor()
    e = construct_two.load_method('euler', 'diffusers.schedulers.scheduling_euler_discrete', 'EulerDiscreteScheduler.from_pretrained')
    scheduler = construct_two.call_method('euler', "/Users/unauthorized/Downloads/models/metadata/sdxl-base/scheduler/scheduler_config.json")

# self._is_available = False
# self._is_built = False
# self._device_count = 0
# self._get_device_name = None
# self._is_flash_attention_available = False
# self._mem_efficient_sdp_enabled = False
# self._enable_attention_slicing = False
# self._max_recommended_memory = 0
# self._max_memory_reserved = 0
