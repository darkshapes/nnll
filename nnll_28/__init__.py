### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import mmap
import pickle


def metadata_from_pickletensor(file_path_named: str) -> dict:
    """
    Collect metadata from a pickletensor file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """
    with open(file_path_named, "r+b") as file_contents_to:
        mm = mmap.mmap(file_contents_to.fileno(), 0)
        return pickle.loads(memoryview(mm))
