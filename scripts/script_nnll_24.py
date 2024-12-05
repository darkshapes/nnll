
from modules.nnll_24.src import find_value_path

reference_map = {  # Example nested dictionary and target values
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

file_tags = {'c': 2, 'd': 1}

matching_path = find_value_path(reference_map, file_tags)  # Find the matching path

print("Matching path:", matching_path)
