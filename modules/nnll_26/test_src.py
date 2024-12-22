#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import unittest
import torch

from modules.nnll_26.src import random_tensor


class TestRandomFunctions(unittest.TestCase):

    def setUp(self):
        # Reset seeds before each test
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(0)
        if torch.xpu.is_available():
            torch.xpu.manual_seed(0)

    def tearDown(self):
        # Reset seeds after each test (if needed)
        pass

    def test_random_tensor(self):
        # Test seeded case
        seed = 12345
        random_tensor(input_seed=seed)  # Seed the RNG
        value1 = torch.rand(1).item()
        random_tensor(input_seed=seed)  # Seed the RNG again to ensure the same state
        value2 = torch.rand(1).item()
        self.assertEqual(value1, value2)

        # Test unseeded case
        random_tensor()  # No seed provided, so it uses a different state each time
        value1 = torch.rand(1).item()
        random_tensor()  # Again, no seed provided
        value2 = torch.rand(1).item()
        self.assertNotEqual(value1, value2)
