
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import os
import hashlib

def compute_file_hash(file_path_named: str) -> str:
    """
    Compute and return the SHA256 hash of a given file.\n
    :param file_path_named: `str` Valid path to a file
    :return: `str` Hexadecimal representation of the SHA256 hash.
    :raises FileNotFoundError: File does not exist at the specified path.
    :raises PermissionError: Insufficient permissions to read the file.
    :raises IOError:  I/O related errors during file operations.
    """
    if not os.path.exists(file_path_named):
        raise FileNotFoundError(f"File '{file_path_named}' does not exist.")
    else:
        with open(file_path_named, 'rb') as file_to_hash:
            return hashlib.sha256(file_to_hash.read()).hexdigest()
