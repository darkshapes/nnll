
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest

import os
import sys
from modules.nnll_23.src import DynamicMethodConstructor


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


if __name__ == "__main__":

    constructor = DynamicMethodConstructor()
    # Load methods dynamically based on system specifications or available files
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
