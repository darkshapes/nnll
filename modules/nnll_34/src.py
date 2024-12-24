#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import re
import os
import pathlib as pl
from collections import defaultdict

from modules.nnll_35.src import capture_title_numeral
from modules.nnll_32.src import get_model_header

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
        else:
            return None, None, None

def gather_sharded_files(file_path: str) -> list:
    """
    Check if a file is sharded, and if so return it\n
    :param file_name: `str` The file name to inspect
    :return: `tuple` of `str` The matching sequence separated into individual variables
    """
    shard_list = []
    file_dir = pl.Path(file_path).parts
    filename = pl.Path(file_path).name # take one at a time, and only the basename/tail
    part, sep, total = detect_index_sequence(filename) # split the sequence numbers from the filename string
    if part is not None and sep is not None and total is not None: # make sure these strings exist
        high_shard = int(total) # translate strings to numbers
        current_shard = int(part)
        shard_list.append(filename)
        for i in range(1,high_shard+1): #compare the numbers to get the file names we need
            if i == current_shard:
                next
            else:
                numeric_to_replace = str(current_shard) # the ceiling
                new_numeric = part.replace(numeric_to_replace, str(i)) + sep
                new_filename = new_numeric.join(filename.split(part+sep,1))
                s = os.sep
                file_dir = os.path.normpath(s.join(file_dir))
                if os.path.exists(os.path.join(file_dir,new_filename)):
                    shard_list.append(new_filename)
                    file_prefix = next(iter(filename.split(part)))
                    file_paths_shard_linked = file_path.remove(new_filename)
                else:
                    return None
                    break
    else:
        return [file_path]

    return file_paths_shard_linked


    #if Path(file_paths).is_dir() == True: #if we are working with a directory
        #for each_file in os.listdir(file_paths): # collect all the files

        #do processing of the list herehere

    #             capture_title_numeral(each_file.name, )
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
    #