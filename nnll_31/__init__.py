### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def compare_layers(query: str, file_path: str) -> None:
    """
    Compare search string with text files of model layers
    Print matches to console\n
    :param query: Search term to scan for
    :param file_path: The folder of JSON files to scan
    """
    from nnll_01.__init__ import nfo
    from nnll_30 import read_json_file
    import json

    try:
        content = read_json_file(file_path)  # Split into lines for easier processing
    except json.JSONDecodeError as error_log:
        nfo("JSON Decode failed %s", f"'{file_path}' invalid JSON format.", error_log)
    except IOError as error_log:
        nfo(f"Error reading file {file_path}: {error_log}")
    else:
        # Find the line containing the pattern
        match_line = next((line for line in content if query in line), None)

        if match_line:
            tensor_count = len(content)
            try:
                # Attempt to parse the line as JSON

                match_key = next({"layer": x, "details": y} for x, y in content.items() if x in match_line)
                if not isinstance(match_key["details"], dict):
                    table_entries = {"shape": match_key["details"], "tensors": tensor_count}
                else:
                    table_entries = {"shape": match_key["details"].get("shape"), "tensors": tensor_count}  # "shapes": shape_value,}
            except json.JSONDecodeError as error_log:
                nfo("Warning: Line containing %s", f"'{query}' in {file_path} is not valid JSON.", error_log)
            else:
                nfo(file_path, table_entries)  # output information


def main():
    """
    Find pattern matches using recursive `grep` search\n
    If 'grep' is not available, only the current folder is scanned.
    """
    import argparse
    import subprocess
    from nnll_01.__init__ import nfo

    parser = argparse.ArgumentParser(description="Search layer name metadata in the current folder's models.")
    parser.add_argument("pattern", help="Pattern to search for")
    args = parser.parse_args()

    query = str(args.pattern)
    try:
        result = subprocess.run(["grep", "-Rl", query], capture_output=True, text=True, check=False)
        files_with_pattern = result.stdout.splitlines()
    except FileNotFoundError:
        nfo("Error: 'grep' command not found.")
        import os

        files_with_pattern = os.listdir(os.getcwd())

    if files_with_pattern:
        for file_path in files_with_pattern:
            compare_layers(query, file_path)
    else:
        nfo(f"No files containing '{query}' were found.")


if __name__ == "__main__":
    main()
