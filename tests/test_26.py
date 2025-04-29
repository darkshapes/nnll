# # // SPDX-License-Identifier: LAL-1.3
# # // d a r k s h a p e s

# import unittest
# import torch
# import torch.xpu as xpu
# from nnll_26 import random_tensor_from_gpu


# class TestRandomFunctions(unittest.TestCase):
#     def setUp(self):
#         # Reset seeds before each test
#         torch.manual_seed(0)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(0)
#         if torch.backends.mps.is_available():
#             torch.mps.manual_seed(0)
#         if xpu.is_available():
#             xpu.manual_seed(0)

#     def tearDown(self):
#         # Reset seeds after each test (if needed)
#         pass

#     def test_random_tensor_from_gpu(self):
#         # Test seeded case
#         seed = 12345
#         random_tensor_from_gpu(input_seed=seed)  # Seed the RNG
#         value1 = torch.rand(1).item()
#         random_tensor_from_gpu(input_seed=seed)  # Seed the RNG again to ensure the same state
#         value2 = torch.rand(1).item()
#         self.assertEqual(value1, value2)

#         # Test unseeded case
#         random_tensor_from_gpu()  # No seed provided, so it uses a different state each time
#         value1 = torch.rand(1).item()
#         random_tensor_from_gpu()  # Again, no seed provided
#         value2 = torch.rand(1).item()
#         self.assertNotEqual(value1, value2)
