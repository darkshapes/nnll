

import sys
import os
import json

from modules.nnll_30.src import read_json_file


def count_tensors_and_extract_shape(pattern, file_path):
    try:
        content = read_json_file(file_path)  # Split into lines for easier processing

        # Find the line containing the pattern
        match_line = next((line for line in content if pattern in line), None)

        if match_line:
            file_location = os.path.join(os.path.dirname("."), match_line)
            count = len(content)
            try:
                # Attempt to parse the line as JSON
                match_key = next([x, y] for x, y in content.items() if x in match_line)
                shape_value = str(content[match_key[0]].get("shape"))
                table_entries = {"shapes": shape_value, "tensors": count}
            except json.JSONDecodeError:
                print(f"Warning: Line containing '{pattern}' in {file_path} is not valid JSON.")
            else:
                print(file_path, table_entries)  # output information

       # else:
            # print(f"No line containing '{pattern}' found in {file_path}.")
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
