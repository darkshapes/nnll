
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

from modules.nnll_32.src import get_model_header
from modules.nnll_07.src import Domain, Architecture, Component
from modules.nnll_27.src import pretty_tabled_output
from modules.nnll_29.src import BlockScanner
from modules.nnll_30.src import read_json_file, write_json_file


def parse_model_header(model_header: dict, filter_file="./filter.json") -> dict:
    try:  # Be sure theres something in model_header
        next(iter(model_header))
    except TypeError as errorlog:
        raise  # Fail if header arrives empty, rather than proceed
    else:  # Process and output metadata
        FILTER = read_json_file(filter_file)
        tensor_count = len(model_header)
        block_scan = BlockScanner()
        file_metadata = block_scan.filter_metadata(FILTER, model_header, tensor_count)
        return file_metadata


def create_model_tag(file_metadata: dict) -> dict:

    if "unknown" in file_metadata:
        domain_dev = Domain("dev")  # For unrecognized models,
    else:
        domain_ml = Domain("ml")  # create the domain only when we know its a model

    arch_found = Architecture(file_metadata.get("model"))
    category = file_metadata["category"]
    file_metadata.pop("model")
    file_metadata.pop("category")
    comp_inside = Component(category, **file_metadata)
    arch_found.add_component(comp_inside.model_type, comp_inside)
    domain_ml.add_architecture(arch_found.architecture, arch_found)
    index_tag = domain_ml.to_dict()
    return index_tag


def prepare_tags(disk_path: str) -> None:
    data = get_model_header(disk_path)  # save_location)
    if data is not None:
        model_header, disk_size, file_name, file_extension = data
    else:
        return
    parse_file = parse_model_header(model_header)
    attribute_dict = {"disk_size": disk_size, "disk_path": disk_path, "file_name": file_name, "file_extension": file_extension}
    file_metadata = parse_file | attribute_dict
    index_tag = create_model_tag(file_metadata)
    try:
        pretty_tabled_output(next(iter(index_tag)), index_tag[next(iter(index_tag))])  # output information
    except TypeError as errorlog:
        raise
    else:
        return index_tag


file_path = "/Users/unauthorized/Downloads/models/image"
save_location = "/Users/unauthorized/Downloads/models/metadata"
index = defaultdict(dict)

if Path(file_path).is_dir() == True:
    path_data = os.listdir(file_path)
    print("\n\n\n\n")
    for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
        file = os.path.join(file_path, each_file)
        index_tag = prepare_tags(file)
        if index_tag is not None:
            index.setdefault(file, index_tag)

elif Path(file_path).exists:
    index = prepare_tags(file_path)

if index is not None and index != {}:
    write_json_file(save_location, "index.json", index, 'a')

# if __name__ == "__main__":
#     main()
# else:
#     file_path = sys.argv[1]
    # Send index_tag data to .json file

# import argparse
# def main(re_file: str = None, re_save: str = None) -> None:
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Analyze model files in a directory and ouptut result to console and json file.")
#     parser.add_argument(
#         "--path", type=str, help="Path to directory or file to be analyzed."
#     )
#     parser.add_argument(
#         "--save", type=str, help="Location to save a .json of output."
#     )
#     args = parser.parse_args()
#     # Call the run function with arguments from flags
#     # 重みを確認するモデルファイル
#     if re_file is None and re_save is None:
#         files = args.path
#         save_location = args.save
#     else:
#         files = prompt
#     if save_location is None or Path(save_location).exists == False:
#         prompt = input("Invalid save location, try again, or press Enter to exit.")
#         if prompt is None:
#             return
#         main(re_file=files, re_save=prompt)
#         return
#     elif files is None or Path(files).exists == False:
#         prompt = input("Invalid scan location, try again, or press Enter to exit.")
#         if prompt is None:
#             return
#         main(re_file=prompt, re_save=save_location)
#         return


#     else:
# #         return
# if len(sys.argv) == 1:
#     # file_path = "/Users/unauthorized/Downloads/models/image/auraflow.diffusers.1of2.fp16.safetensors"
#     # file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"

# file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"
