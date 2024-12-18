
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


# Example usage
v = {
    'filename': 'image01.png',
    'size': '123456',
    'date': '2023-10-01'
}

file_tags = {
    'filename': "r'image\d+.png'",  # Regular expression for filenames like image01.png, image02.png, etc.
    'size': '123456',
    'date': '2023-10-01'
}

result = find_value_path(v, file_tags)
print(result)  # Output: True
