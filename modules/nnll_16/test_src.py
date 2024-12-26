
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

# import unittest
# from unittest.mock import patch, MagicMock
# from abc import ABC, abstractmethod
# import torch
# import os
# import sys
# from modules.nnll_17.src import Backend, CUDADevice, MPSDevice, XPUDevice

# # Mocking torch module for testing
# mock_torch = MagicMock()
# mock_torch.cuda.is_available.return_value = True
# mock_torch.backends.cuda.is_built.return_value = True
# mock_torch.cuda.device_count.return_value = 2
# mock_torch.get_device_properties = lambda device: MagicMock(total_memory=16 * 1024**3)
# mock_torch.cuda.get_device_name = lambda idx: f"MockDevice{idx}"
# mock_torch.is_flash_attention_available.return_value = False
# mock_torch.mem_efficient_sdp_enabled.return_value = True

# mock_torch.mps.is_available.return_value = False
# mock_torch.backends.mps.is_built.return_value = False
# mock_torch.mps.recommended_max_memory.return_value = 8 * 1024**3
# mock_torch.mps.enable_attention_slicing.return_value = None

# mock_torch.xpu.is_available.return_value = True
# mock_torch.xpu.device_count.return_value = 1
# mock_torch.xpu.max_memory_reserved.return_value = 4 * 1024**3
# mock_torch.xpu.get_device_name.return_value = "MockXPUDevice"

# # Patch torch with our mock object
# patcher = patch.dict('sys.modules', {'torch': mock_torch})
# patcher.start()


# class TestBackend(ABC):
#     @abstractmethod
#     def setUp(self):
#         pass

#     def test_configure_called(self):
#         self.backend.configure()
#         self.assertTrue(hasattr(self.backend, '_is_available'))
#         self.assertTrue(hasattr(self.backend, '_is_built'))

#     def test_attribute_method_exists(self):
#         result = self.backend.attribute('is_available')
#         self.assertIsNotNone(result)
#         print(result)
#         self.assertIsNotNone(result, mock_torch.cuda.is_available.return_value)

#     def test_device_count(self):
#         device_count = self.backend.attribute('device_count')
#         self.assertIsNotNone(device_count)
#         self.assertEqual(device_count, mock_torch.xpu.device_count.return_value if isinstance(self.backend, XPUDevice) else mock_torch.cuda.device_count.return_value)


# class TestCUDADevice(unittest.TestCase, TestBackend):
#     def setUp(self):
#         self.backend = CUDADevice()

#     def test_specific_cuda_attributes(self):
#         self.assertTrue(hasattr(self.backend, '_get_device_name'))
#         self.assertEqual(len(self.backend._get_device_name), 2)
#         self.assertEqual(self.backend._is_flash_attention_available, False)
#         self.assertEqual(self.backend._mem_efficient_sdp_enabled, True)


# class TestMPSDevice(unittest.TestCase, TestBackend):
#     def setUp(self):
#         self.backend = MPSDevice()

#     def test_specific_mps_attributes(self):
#         self.assertFalse(hasattr(self.backend, '_get_device_name'))
#         self.assertEqual(self.backend._recommended_max_memory, 8 * 1024**3)


# class TestXPUDevice(unittest.TestCase, TestBackend):
#     def setUp(self):
#         self.backend = XPUDevice()

#     def test_specific_xpu_attributes(self):
#         self.assertTrue(hasattr(self.backend, '_get_device_name'))
#         self.assertEqual(len(self.backend._get_device_name), 1)
#         self.assertEqual(self.backend._max_memory_reserved, 4 * 1024**3)
