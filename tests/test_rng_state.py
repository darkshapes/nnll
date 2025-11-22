# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Tests for RNGState class to verify random number generation, seed storage, and device handling."""

import pytest
import torch
from nnll.random import RNGState


class TestRNGState:
    """Test suite for RNGState class."""

    def test_generates_random_numbers(self):
        """Test that RNGState can generate random numbers via torch generator."""
        seed_state = RNGState()

        gen = seed_state._torch_generator
        results = [torch.rand(1, generator=gen).item() for _ in range(100)]
        assert len(set(results)) > 1, "RNGState should produce different random values"
        for result in results:
            assert 0.0 <= result < 1.0, f"Result {result} out of expected range [0, 1)"

    def test_stores_seed_correctly(self):
        """Test that RNGState stores seeds correctly."""
        seed = 42
        seed_state = RNGState(initial_seed=seed)
        assert seed_state.seed == seed, "Seed should be stored correctly"
        new_seed = 123
        seed_state.set_seed(new_seed)
        assert seed_state.seed == new_seed, "New seed should be stored correctly"

    def test_stores_seed_when_none_provided(self):
        """Test that RNGState generates and stores a seed when None is provided."""
        seed_state = RNGState()
        assert seed_state.seed is None, "Initial seed should be None when not provided"
        generated_seed = seed_state.set_seed()
        assert seed_state.seed is not None, "Seed should be generated and stored"
        assert isinstance(seed_state.seed, int), "Stored seed should be an integer"
        assert seed_state.seed == generated_seed, "Returned seed should match stored seed"

    def test_set_device_changes_generator(self):
        """Test that set_device properly updates the generator device."""
        seed_state = RNGState(device="cpu")
        assert seed_state.device == "cpu"
        assert seed_state._torch_generator.device.type == "cpu"
        if torch.cuda.is_available():
            seed_state.set_device("cuda")
            assert seed_state.device == "cuda", "Device should be updated"
            assert seed_state._torch_generator.device.type == "cuda", "Generator should be on new device"
        elif torch.backends.mps.is_available():
            seed_state.set_device("mps")
            assert seed_state.device == "mps", "Device should be updated"
            assert seed_state._torch_generator.device.type == "mps", "Generator should be on new device"

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces deterministic results."""
        seed = 42
        seed_state1 = RNGState(initial_seed=seed)
        seed_state2 = RNGState(initial_seed=seed)
        gen1 = seed_state1._torch_generator
        gen2 = seed_state2._torch_generator
        results1 = [torch.rand(1, generator=gen1).item() for _ in range(10)]
        results2 = [torch.rand(1, generator=gen2).item() for _ in range(10)]
        assert results1 == results2, "Same seed should produce identical results"

    def test_seed_propagates_to_torch_global_state(self):
        """Test that set_seed properly seeds torch global RNG state."""
        seed = 999
        seed_state = RNGState()
        seed_state.set_seed(seed)
        torch.manual_seed(seed)
        global_result1 = torch.rand(1).item()
        torch.manual_seed(seed)
        global_result2 = torch.rand(1).item()

        assert global_result1 == global_result2, "Global torch RNG should be seeded correctly"

    def test_devices_seeding_when_available(self):
        """Test that CUDA devices are seeded when available."""
        if torch.cuda.is_available():
            seed = 12345
            seed_state = RNGState(device="cuda")
            seed_state.set_seed(seed)
            cuda_result1 = torch.rand(1, device="cuda").item()
            seed_state.set_seed(seed)
            cuda_result2 = torch.rand(1, device="cuda").item()

            assert cuda_result1 == cuda_result2, "CUDA RNG should be seeded correctly"

        if torch.backends.mps.is_available():
            seed = 54321
            seed_state = RNGState(device="mps")
            seed_state.set_seed(seed)
            mps_result1 = torch.rand(1, device="mps").item()
            seed_state.set_seed(seed)
            mps_result2 = torch.rand(1, device="mps").item()

            assert mps_result1 == mps_result2, "MPS RNG should be seeded correctly"


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
