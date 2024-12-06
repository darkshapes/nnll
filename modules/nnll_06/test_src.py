

import pytest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_06.src import compare_dicts


class TestCompareDict:
    def test_value_comparison(self):
        known_values = {
            "general": {
                "architecture": "LLaMA-7B",
                "name": "LLaMA-7B"
            },
            "dtype": "float32"
        }

        metadata = {
            "general": {
                "architecture": "LLaMA-7B",
                "name": "LLaMA-7B",
                "version": "1.0"  # This key is not in known_values and should be ignored
            },
            "dtype": "float32",
            "additional_key": "some_value"  # This key is not in known_values and should be ignored
        }

        metadata_incorrect = {
            "general": {
                "architecture": "CCaML-12B",
                "name": "CCaML-12B",
                "version": "1.0"  # This key is not in known_values and should be ignored
            },
            "dtype": "float32",
            "additional_key": "some_value"  # This key is not in known_values and should be ignored
        }

        assert compare_dicts(known_values, metadata) == True
        assert compare_dicts(known_values, metadata_incorrect) == False
