import re
from pathlib import Path

def extract_part_and_total(filename: str) -> tuple:
    """
    Extracts the part number and total parts from the filename.
    Supports formats like 00001-of-00003, 1of3, etc.
    """
    # rewrite this above
    patterns = [ r'(\d+)(-[oO][fF]-)(\d+)', r'(\d)([oO][fF])(\d)'] # at least one number, hyphen of hyphen number, with or without hyphen, | previously # r'[0000](\d+)[oO][fF][0000](\d+)'
    for pattern in patterns:
        expression = re.compile(pattern)
        match = re.search(expression, filename)
        if match:
            part, sep, total = map(str, match.groups())
            return part, sep, total
    return None, None

def process_files(file_paths: list) -> list:
    """
    Processes a list of file paths in the correct order based on their naming convention.
    Returns a list of results from get_model_header for each file.
    """
    # placeholder, rewrite above
    # Create a dictionary to hold files and their part numbers
    files_dict = {}
    for file_path in file_paths:
        part, sep, total = extract_part_and_total(Path(file_path).name)
        if part is not None and total is not None:
            next_path = file_path.replace((part + sep + total), (total + sep + total))
            if total not in files_dict:
                files_dict[total] = {}
            files_dict[total][part] = file_path

    # Process each group of files in the correct order
    results = {}
    for total, parts in sorted(files_dict.items()):
        for part in sorted(parts):

            #result = get_model_header(parts[part])
            if result:
                results = result | results.copy()

    return results

file_paths = [
    'model_00001-of-00003.safetensors',
    'model_00002-of-00003.safetensors',
    'model_00003-of-00003.safetensors',
    'model_00001-of-00002.gzip',
    'model_00002-of-00002.gzip',
    'model_1of3.tar.gz',
    'model_2of3.tar.gz',
    'model_3of3.tar.gz'
]

results = process_files(file_paths)
for result in results:
    print(result)
    #