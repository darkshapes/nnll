# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import os
import hashlib
from huggingface_hub import get_hf_file_metadata, hf_hub_url


def retrieve_remote_hash(repo_id: str, file_path_absolute: str):
    """Return the hash value of a file from HF"""
    from_remote_location = hf_hub_url(repo_id=repo_id, filename=file_path_absolute)
    metadata = get_hf_file_metadata(from_remote_location)
    return metadata.etag


def collect_hashes(hash_dict):
    """Arrange files for hash retrieval"""
    hash_values = {}
    for file in hash_dict:
        if os.path.isfile(file):
            hash_values.setdefault(file, compute_hash_for(file))
    return hash_values


def compute_hash_for(file_path_named: str) -> str:
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
        with open(file_path_named, "rb") as file_to_hash:
            return hashlib.sha256(file_to_hash.read()).hexdigest()
