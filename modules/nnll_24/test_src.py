
import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from nnll_24.src import find_value_path


def test_find_value_path():
    reference_map = {
        'x': {
            'blocks': 'c1d2',
            'shapes': 256
        },
        'z': {
            'blocks': "c2d2",
            'shapes': 512
        },
        'y': {
            'blocks': "c.d",
            'shapes': 256
        }
    }

    # Test matching path
    file_tags = {
        'c.d': {
            "shape": 256 }
    }
    print(find_value_path(reference_map, file_tags))
    assert find_value_path(reference_map, file_tags) == ["y"]

    # Test no match found
    file_tags_no_match = {'e': 3, 'f': 4}
    assert find_value_path(reference_map, file_tags_no_match) is None

    # Test deeper nested structure
    reference_map_deeper = {
        'level1': {
            'level2': {
                'level3': {
                    'blocks': 'c.d',
                    'shape': 256
                }
            },
            'another': {'skip': {}}
        }
    }
    print(find_value_path(reference_map_deeper, file_tags))
    assert find_value_path(reference_map_deeper, file_tags) == ['level1', 'level2', 'level3']

    # Test with empty dict
    reference_map_empty = {}
    assert find_value_path(reference_map_empty, file_tags) is None

    # Test matching at the top level
    reference_map_top_level_match = {'c.d': { "shape": 256} }
    assert find_value_path(reference_map_top_level_match, file_tags) == ['c.d']


test_find_value_path()
