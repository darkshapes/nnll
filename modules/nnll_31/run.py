
import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.pardir)))
from nnll_31.src import count_tensors_and_extract_shape


def find_files_with_pattern(pattern):
    if not pattern:
        print(f"Usage: {sys.argv[0]} <pattern>")
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pattern>")
        sys.exit(1)

    pattern = sys.argv[1]
    find_files_with_pattern(pattern)
