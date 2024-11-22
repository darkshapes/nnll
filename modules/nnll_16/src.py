
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from functools import reduce

class Backend(ABC):
    """Dynamically retrieve methods and set attribute flags given a library."""
    def __init__(self, module_name, backend_type: str):
        """
        #### Class structured around iterating methods from a framework
        #### `backend_type`: device suffix received from subclasses
        #### OUTPUT:
        """

        self.framework = lambda module_name: import_module(module_name) if module_name in sys.modules else None
        self.backend_type = backend_type
        self.configure()

    @abstractmethod
    def configure(self):
        pass

    def attribute(self, methods: list) -> any:
        """
        #### Dynamically set class variables based on system configuration
        #### `method`: The sequence of methods as a list
        #### `framework`: The base framework for the methods (default is Pytorch)
        #### OUTPUT: Boolean values assigned to respective main class variables
        """
        try:
            method = reduce(lambda f, m: getattr(f, m), methods, self.framework)
            result = method()
            setattr(self, f"_{methods[:-1]}", result)
            return result
        except AttributeError as e:
            print(f"AttributeError: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
