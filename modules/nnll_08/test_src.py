
import unittest
import secrets
from numpy.random import SeedSequence, Generator, Philox
from torch import torch

# Import your functions
from src import soft_random, hard_random, tensor_random, seed_planter


class TestRandomFunctions(unittest.TestCase):

    def test_soft_random(self):
        # Test deterministic behavior with a fixed seed
        entropy = f"0x{secrets.randbits(128):x}"
        rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
        expected_value = int(rndmc.integers(0, 0x2540BE3FF))

        # Set the seed for soft_random to match
        with unittest.mock.patch('secrets.randbits', return_value=int(entropy, 16)):
            self.assertEqual(soft_random(), expected_value)

        # Test range of values
        value = soft_random()
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 0x2540BE3FF)

    def test_hard_random(self):
        # Test non-deterministic behavior
        value1 = hard_random(hardness=5)
        value2 = hard_random(hardness=5)
        self.assertNotEqual(value1, value2)

        # Test length of the returned number based on hardness
        for hardness in range(1, 6):
            value = hard_random(hardness)
            expected_length = hardness * 2  # Each hex digit is 4 bits
            self.assertEqual(len(hex(value)), expected_length + 2)  # +2 for '0x' prefix

    def test_tensor_random(self):
        # Test seeded case
        seed = 12345
        tensor_random(seed)
        value1 = torch.rand(1).item()
        tensor_random(seed)
        value2 = torch.rand(1).item()
        self.assertEqual(value1, value2)

        # Test unseeded case
        tensor_random()
        value1 = torch.rand(1).item()
        tensor_random()
        value2 = torch.rand(1).item()
        self.assertNotEqual(value1, value2)

    def test_seed_planter(self):
        seed = 12345

        # Test with CUDA available
        if torch.cuda.is_available():
            config = seed_planter(seed, deterministic=True)
            self.assertEqual(config['torch.backends.cudnn.deterministic'], 'True')
            self.assertEqual(config['torch.backends.cudnn.benchmark'], 'False')

            # Check that seeds are set correctly
            self.assertEqual(torch.initial_seed(), seed)
            if torch.cuda.device_count() > 0:
                self.assertEqual(torch.cuda.initial_seed(), seed)

        # Test with MPS available
        elif torch.backends.mps.is_available():
            config = seed_planter(seed, deterministic=True)
            self.assertIsNone(config)  # No specific return value for MPS

            # Check that seeds are set correctly
            self.assertEqual(torch.initial_seed(), seed)

        # Test without CUDA or MPS
        else:
            config = seed_planter(seed, deterministic=True)
            self.assertIsNone(config)  # No specific return value

            # Check that seeds are set correctly
            self.assertEqual(torch.initial_seed(), seed)


if __name__ == '__main__':
    unittest.main()
