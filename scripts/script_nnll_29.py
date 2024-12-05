
import os
from os import Path
from tqdm.auto import tqdm
from modules.nnll_29.src import BlockIndex


models = "/Users/unauthorized/Downloads/models/"
files = save_location = os.path.join(models, "text")
blocks = BlockIndex()
save_location = os.path.join(models, "metadata")
if Path(files).is_dir() == True:
    path_data = os.listdir(files)
    print("\n\n\n\n")
    for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
        file_path = os.path.join(files, each_file)
        blocks.main(file_path, save_location)
else:
    blocks.main(files, save_location)
