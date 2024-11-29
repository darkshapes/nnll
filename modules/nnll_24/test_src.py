
import pytest
import os
import sys

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if modules_path not in sys.path:
    sys.path.append(modules_path)

from nnll_24.src import find_value_path


def test_find_value_path():
    reference_map = {
        'z': {
            'a': {
                'x': {
                    'b1': {'c': 1, 'd': 2},
                    'b2': {'c': 2, 'd': 2},
                },
                'y': {
                    'b': {'c': 2, 'd': 1}
                }
            }
        }
    }

    # Test matching path
    file_tags = {'c': 2, 'd': 1}
    assert find_value_path(reference_map, file_tags) == ['z', 'a', 'y', 'b']

    # Test no match found
    file_tags_no_match = {'e': 3, 'f': 4}
    assert find_value_path(reference_map, file_tags_no_match) is None

    # Test deeper nested structure
    reference_map_deeper = {
        'level1': {
            'level2': {
                'level3': {
                    'target': {'c': 2, 'd': 1},
                    'other': {'e': 5}
                }
            },
            'another': {'skip': {}}
        }
    }

    assert find_value_path(reference_map_deeper, file_tags) == ['level1', 'level2', 'level3', 'target']

    # Test with empty dict
    reference_map_empty = {}
    assert find_value_path(reference_map_empty, file_tags) is None

    # Test matching at the top level
    reference_map_top_level_match = {'c': 2, 'd': 1}
    assert find_value_path(reference_map_top_level_match, file_tags) == ['c', 'd']


if __name__ == "__main__":
    pytest.main([__file__])
