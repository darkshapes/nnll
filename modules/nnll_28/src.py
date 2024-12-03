
import os
import torch

# PICKLETENSOR FILE


def unpickle(self, file_name: str, extension: str):
    self.id_values["file_size"] = os.path.getsize(file_name)
    import mmap
    import pickle
    try:
        return torch.load(file_name, map_location="cpu")  # this method seems outdated
    except TypeError as error_log:
        self.error_handler(kind="retry", error_log=error_log, obj_name=file_name, error_source=extension)
        try:
            with open(file_name, "r+b") as file_obj:
                mm = mmap.mmap(file_obj.fileno(), 0)
                return pickle.loads(memoryview(mm))
        except Exception as error_log:  # throws a _pickle error (so salty...)
            self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension)
