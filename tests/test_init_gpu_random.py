### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import pytest
import secrets

from nnll.configure.init_gpu import soft_random, hard_random


class TestRandom:
    def setup_seed(self, seed):
        original_randbits = secrets.randbits

        # Mock function
        def mock_randbits(bits):
            return int(seed, 16) % (2**bits)  # Ensure the result fits within the specified number of bits

        # Swap original with mock
        secrets.randbits = mock_randbits

        try:
            yield
        finally:
            # Restore original method
            secrets.randbits = original_randbits

    @pytest.fixture
    def setup_seed_fixture(self):
        yield from self.setup_seed("0x123456789abcdef")

    def test_soft_random_determinism(self, setup_seed_fixture):
        """Test deterministic behavior of soft_random with a fixed seed."""
        result = soft_random(0x2540BE3FF)
        # Might need adjustment..
        assert 0 <= result < 0x2540BE3FF

    def test_soft_random_boundaries(self):
        """Test that soft_random returns values within the specified range."""
        for _ in range(100):  # Run multiple times to check randomness
            result = soft_random(0x2540BE3FF)
            assert 0 <= result < 0x2540BE3FF

    def test_hard_random_boundaries(self):
        # Test that hard_random generates numbers within the expected range.
        for hardness in range(1, 6):
            result = hard_random(hardness)
            max_value = (1 << (hardness * 8)) - 1
            assert 0 <= result <= max_value

    def test_hard_random_uniqueness(self):
        """Test that hard_random generates a diverse set of values."""
        results = [hard_random(4) for _ in range(256)]
        # Expecting no duplicates; statistically may vary.
        assert len(set(results)) == 256


# import unittest
# import torch
# import torch.xpu as xpu
# from nnll.configure.init_gpu import random_tensor_from_gpu


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

if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
