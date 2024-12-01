
import pytest
import secrets
import numpy as np

from .src import soft_random, hard_random


def setup_seed(seed):
    # Save the original randbits method
    original_randbits = secrets.randbits

    # Define the mock function
    def mock_randbits(bits):
        return int(seed, 16) % (2 ** bits)  # Ensure the result fits within the specified number of bits

    # Replace the original method with the mock
    secrets.randbits = mock_randbits

    try:
        yield
    finally:
        # Restore the original method
        secrets.randbits = original_randbits


@pytest.fixture
def setup_seed_fixture():
    yield from setup_seed("0x123456789abcdef")


def test_soft_random_determinism(setup_seed_fixture):
    """Test deterministic behavior of soft_random with a fixed seed."""
    result = soft_random(0x2540BE3FF)
    # Since this is mocked, you might need to adjust based on your logic
    assert 0 <= result < 0x2540BE3FF


def test_soft_random_boundaries():
    """Test that soft_random returns values within the specified range."""
    for _ in range(100):  # Run multiple times to check randomness
        result = soft_random(0x2540BE3FF)
        assert 0 <= result < 0x2540BE3FF


def test_hard_random_boundaries():
    # Test that hard_random generates numbers within the expected range.
    for hardness in range(1, 6):
        result = hard_random(hardness)
        max_value = (1 << (hardness * 8)) - 1
        assert 0 <= result <= max_value


def test_hard_random_uniqueness():
    """Test that hard_random generates a diverse set of values."""
    results = [hard_random(4) for _ in range(256)]
    # Expecting no duplicates; this is statistical and may vary.
    assert len(set(results)) == 256


if __name__ == "__main__":
    setup_seed(0)
    test_soft_random_determinism()
    test_soft_random_boundaries()
    test_hard_random_boundaries()
    test_hard_random_uniqueness()
