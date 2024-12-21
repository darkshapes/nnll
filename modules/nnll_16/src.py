#// SPDX-License-Identifier: MIT
#// d a r k s h a p e s

import sys
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TypeVar, Callable
from functools import reduce

ancestor = TypeVar("ancestor", covariant=True)


class Backend(ABC):
    """Dynamically retrieve methods and set attribute flags given a library."""

    def __init__(self, module_name: str = None, package_name: Callable = None):
        """
        #### Class structured around iterating methods from a framework
        #### `package_name`: base
        #### `module_name`: device suffix received from subclasses
        #### OUTPUT:
        """
        self.module_name = module_name
        self.framework = lambda package_name: import_module(package_name) if package_name in sys.modules else None
        self.configure()

    @abstractmethod
    def configure(self):
        pass

    def attribute(self, methods: list) -> any:
        """
        #### Dynamically set class variables
        #### `method`: The sequence of methods as a list
        #### `framework`: The base framework for the methods
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
