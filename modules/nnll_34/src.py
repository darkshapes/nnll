
import re
import os
from pathlib import Path

from modules.nnll_35.src import get_count_from_filename

def capture_sequence_index(file_name: str) -> tuple:
    """
    Find a number pair in a file name and return the sequence, including characters between it.\n
    :param file_name: `str` The file name to inspect
    :return: `tuple` of `str` The matching sequence separated into individual variables
    """
    patterns = [ r'(\d+)(-[oO][fF]-)(\d+)', r'(\d)([oO][fF])(\d)'] # at least one number, hyphen of hyphen number, with or without hyphen, | previously # r'[0000](\d+)[oO][fF][0000](\d+)'
    for pattern in patterns:
        expression = re.compile(pattern)
        match = re.search(expression, file_name)
        if match:
            part, sep, total = map(str, match.groups())
            return part, sep, total
    return None, None, None

def preprocess_files(file_paths: list) -> list:
    """
    Evaluate and direct model and sharded model file loading.\n

    """
    # """ indicates a sequence of files
    # Processes a list of file paths in the correct order based on their naming convention.
    # Returns a list of results from get_model_header for each file.
    # """
    # # placeholder, rewrite above
    # # Create a dictionary to hold files and their part numbers

    if Path(file_paths).is_dir() == True: #if we are working with a directory
        for each_file in os.listdir(file_paths):
            filename = Path(each_file).name
            part, sep, total = capture_sequence_index(filename)
            if part is not None and sep is not None and total is not None:
                high_shard = int(total)
                current_shard = int(part)
                shard_list = []
                for i in range(1,high_shard):
                    if i == current_shard:
                        next
                    else:
                        numeric_to_replace = str(current_shard)
                        numeric_to_replace_with = str(high_shard)
                        new_numeric = total.replace(numeric_to_replace_with, str(i))

                        shard_list.extend(new_numeric.join(filename.rsplit(numeric_to_replace, 1)))


    #             get_count_from_filename(each_file.name, )
    #             next_path = file_path.replace((part + sep + total), (total + sep + total))
    #             if total not in files_dict:
    #                 files_dict[total] = {}
    #             files_dict[total][part] = file_path

    #     file = os.path.join(file_paths, each_file)



    # # Process each group of files in the correct order
    # results = {}
    # for total, parts in sorted(files_dict.items()):
    #     for part in sorted(parts):

    #         #result = get_model_header(parts[part])
    #         if result:
    #             results = result | results.copy()

    # return results

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