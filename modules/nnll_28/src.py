#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


import mmap
import pickle


def load_pickletensor_metadata_from_model(file_path: str) -> dict:
    """
    Collect metadata from a pickletensor file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """
    with open(file_path, "r+b") as file_obj:
        mm = mmap.mmap(file_obj.fileno(), 0)
        return pickle.loads(memoryview(mm))
