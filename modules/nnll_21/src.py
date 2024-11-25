
import os

def symlinker(true_file, metadata_folder, filename, full_path=False):
        symlink_full_path = os.path.join(metadata_folder, filename)
        try:
            os.remove(symlink_full_path)
        except FileNotFoundError:
            pass
        os.symlink(true_file, symlink_full_path)
        return metadata_folder if full_path == False else symlink_full_path
