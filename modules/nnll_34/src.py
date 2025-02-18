
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import re
import os
import pathlib as pl
from collections import defaultdict

from modules.nnll_35.src import capture_title_numeral
from modules.nnll_32.src import coordinate_header_tools

def detect_index_sequence(file_name: str) -> tuple:
    """
    Check for a number pair in a file name and return the sequence, including characters between it.\n
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
    return file_name

def gather_sharded_files(file_path_named, index_segments: str) -> list:
    """
    Check if a file is sharded, and if so return it\n
    :param file_name: `str` The file name to inspect
    :return: `tuple` of `str` The matching sequence separated into individual variables
    """

    part, sep, total = index_segments
    shard_list = []
    file_dir = pl.Path(file_path_named).parts
    filename = pl.Path(file_path_named).name # take one at a time, and only the basename/tail
    # part, sep, total = detect_index_sequence(filename) # split the sequence numbers from the filename string
    if part is None or sep is None or total is None: # make sure these strings exist
        return [file_path_named]
    else:
        high_shard = int(total) # translate strings to numbers
        current_shard = int(part)
        shard_list.append(file_path_named)
        file_dir = os.sep.join(file_dir[:-1]) # Combine .parts Path minus file
        file_dir = os.path.normpath(file_dir[1:]) # Remove leading slash)
        for i in range(1,high_shard+1): #compare the numbers to get the file names we need
            if i == current_shard:
                next
            else:
                numeric_to_replace = str(current_shard)
                new_numeric = part.replace(numeric_to_replace, str(i)) + sep
                new_file_name = new_numeric.join(filename.split(part+sep,1))

                new_file_path_named = os.path.join(file_dir,new_file_name)
                if os.path.exists(new_file_path_named):
                    shard_list.append(new_file_path_named)
                    # file_prefix = next(iter(filename.split(part)))
                    # file_paths_shard_linked = file_path_named.remove(new_filename)
                else:
                    raise FileNotFoundError(f"Shard for {file_path_named} not found")

    return shard_list

