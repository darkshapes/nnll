
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

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
        :param module_path: Dot-separated string representing the path to the module (e.g., 'os.path')
        :param attribute_path: Dot-separated string representing the path to the method within the module (e.g., 'join')
        :return: `None` enlivens the method_name with the module and attributes requested
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
