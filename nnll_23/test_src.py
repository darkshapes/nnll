
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_23.src import DynamicMethodConstructor


class TestDynamicMethodConstructor(unittest.TestCase):
    def setUp(self):
        self.constructor = DynamicMethodConstructor()

    def test_load_method_successfully(self):
        """Test if a method is successfully loaded and can be called."""
        self.constructor.load_method('path_join', 'os.path', 'join')
        result = self.constructor.call_method('path_join', '/usr/local/bin', 'python3.9')
        self.assertEqual(result, '/usr/local/bin/python3.9')

    def test_load_nonexistent_module(self):
        """Test handling of a non-existent module."""
        with self.assertRaises(RuntimeError) as context:
            self.constructor.load_method('nonexistent_func', 'nonexistent_module', 'func')
        self.assertIn("Failed to load module", str(context.exception))

    def test_load_nonexistent_attribute(self):
        """Test loading an attribute that doesn't exist in the module."""
        with self.assertRaises(RuntimeError) as context:
            self.constructor.load_method('invalid_attr', 'os.path', 'nonexistattr')
        self.assertIn("Failed to access attribute", str(context.exception))

    def test_call_nonexistent_method(self):
        """Test calling a method that was not loaded or does not exist."""
        with self.assertRaises(AttributeError) as context:
            self.constructor.call_method('unloaded_method')
        self.assertEqual(str(context.exception), "Method 'unloaded_method' not found.")

    def test_complex_load_and_call(self):
        """Testing loading and calling a complex, real-world method dynamically."""
        self.constructor.load_method('euler', 'numpy.random', 'randn')  # Using numpy as an example
        result = self.constructor.call_method('euler', 2, 3)  # Example with numpy's randn
        self.assertEqual(result.shape, (2, 3))  # Shape should be (2, 3)
