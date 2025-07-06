# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import List, Dict, Iterable


def retrieve_remote_hash(repo_or_model_id: str, file_path_absolute: str, hf: bool = True):
    """Return the hash value of a file from HF"""

    if hf:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        from_remote_location = hf_hub_url(repo_id=repo_or_model_id, filename=file_path_absolute)
        metadata = get_hf_file_metadata(from_remote_location)
        return metadata.etag
    else:
        hash_values = []
        import requests
        from json import JSONDecodeError

        request: requests.Response = requests.get("https://civitai.com/api/v1/models/" + repo_or_model_id, timeout=(10, 10))
        if request.ok and request.status_code == 200:
            try:
                request_data = request.json()
            except JSONDecodeError:
                return
            model_versions: List[Dict[str, str]] = request_data.get("modelVersions")
            for version in model_versions:
                files_data = version.get("files")
                if files_data:
                    for file_data in files_data:
                        hash_values.append(files_data["hashes"].get("SHA256"))
        return hash_values


async def compute_hash_for(file_path_named: str = None, text_stream: str = None) -> str:
    """
    Compute and return the SHA256 hash of a given file or data. *Provide only ONE argument.*\n
    :param file_path_named:  Valid path to a file
    :param text_stream: Plain text to encode
    :return: Hexadecimal representation of the SHA256 hash.
    :raises FileNotFoundError: File does not exist at the specified path.
    :raises PermissionError: Insufficient permissions to read the file.
    :raises IOError:  I/O related errors during file operations.
    """

    import hashlib
    import os

    if not text_stream and not os.path.exists(file_path_named):
        raise FileNotFoundError(f"File '{file_path_named}' does not exist.")
    elif not text_stream:
        with open(file_path_named, "rb") as file_to_hash:
            return hashlib.sha256(file_to_hash.read()).hexdigest()
    else:
        return hashlib.sha256(str(text_stream).encode(encoding="utf-8")).hexdigest()


async def compute_b3_for(file_path_named: str = None, text_stream: str = None) -> str:
    """
    Compute and return the BLAKE3 hash of a given file or data.\n
    ###  *_Provide ONE argument._*\n
    :param file_path_named:  Valid path to a file
    :param text_stream: Plain text to encode
    :return: Hexadecimal representation of the BLAKE3 hash.
    :raises FileNotFoundError: File does not exist at the specified path.
    :raises PermissionError: Insufficient permissions to read the file.
    :raises IOError:  I/O related errors during file operations.
    """

    from blake3 import blake3
    import os

    if not text_stream and not os.path.exists(file_path_named):
        raise FileNotFoundError(f"File '{file_path_named}' does not exist.")
    elif not text_stream:
        file_hasher = blake3(max_threads=blake3.AUTO)
        file_hasher.update_mmap(file_path_named)
        return file_hasher.hexdigest()
    else:
        return blake3(str(text_stream).encode(encoding="utf-8")).hexdigest()


def collect_hashes(hash_stack: Iterable[str]) -> Dict[str, str]:
    """Arrange files for hash retrieval
    :param hash_stack: The file locations to hash
    :returns: A mapping of filenames to their hash values"""

    import os

    hash_values = {}
    for file in hash_stack:
        if os.path.isfile(file):
            hash_values.setdefault(file, compute_hash_for(file))
    return hash_values
