
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import sys
import os
import json
import subprocess
from contextlib import suppress

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

def find_files_with_pattern(pattern):
    if not pattern:
        print(f"Search for layer name metadata in the current folder's models.\n Usage: {sys.argv[0]} <pattern>")
        sys.exit(1)

    try:
        result = subprocess.run(['grep', '-Rl', pattern], capture_output=True, text=True)
        files_with_pattern = result.stdout.splitlines()
    except FileNotFoundError:
        print("Error: 'grep' command not found. Make sure it is installed.")
        sys.exit(1)

    if files_with_pattern:
        for file in files_with_pattern:
            count_tensors_and_extract_shape(pattern, file)
    else:
        print(f"No files containing '{pattern}' were found.")

def find_entry(pattern=(TypeError)):
    with suppress(TypeError):
        if len(sys.argv) != 2:
            print(f"""
Description: Search layer name metadata in the current folder's models.
Usage: {sys.argv[0]} <pattern>\n""")
            return

    pattern = sys.argv[1]
    find_files_with_pattern(pattern)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"""
                Cannot be none.\n Search layer name metadata in the current folder's models. Usage: {sys.argv[0]} <pattern>""")
        sys.exit(1)

    pattern = sys.argv[1]
    find_files_with_pattern(pattern)
