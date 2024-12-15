
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0]), "modules") ))
from nnll_04.src import load_safetensors_metadata
from nnll_05.src import load_gguf_metadata
from nnll_28.src import load_pickletensor_metadata
from nnll_07.src import Domain, Architecture, Component
from nnll_27.src import pretty_tabled_output
from nnll_29.src import BlockScanner
from nnll_30.src import read_json_file, write_json_file


def get_model_header(file_path: str) -> tuple:

    # todo: refine the load methods to work with sharded files
    method_map = {
        ".safetensors": load_safetensors_metadata,
        ".sft": load_safetensors_metadata,
        ".gguf": load_gguf_metadata,
        ".pt": load_pickletensor_metadata,
        ".pth": load_pickletensor_metadata,
        ".ckpt": load_pickletensor_metadata
    }
    # Get external file metadata
    file_extension = Path(file_path).suffix.lower()
    if file_extension == "" or file_extension is None or file_extension not in method_map:  # Skip this file if we cannot possibly know what it is
        return
    file_name = os.path.basename(file_path)
    disk_size = os.path.getsize(file_path)

    # Retrieve header by method indicated by extension, usually struct unpacking, except for pt files which are memmap
    model_header = method_map[file_extension](file_path)
    return (model_header, disk_size, file_name, file_extension)


def parse_model_header(model_header: dict, filter_file="modules/nnll_29/filter.json") -> dict:
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
    model_header, disk_size, file_name, file_extension = get_model_header(disk_path)  # save_location)
    parse_file = parse_model_header(model_header)
    attribute_dict = {"disk_size": disk_size, "disk_path": disk_path, "file_name": file_name, "file_extension": file_extension}
    file_metadata = parse_file | attribute_dict
    index_tag = create_model_tag(file_metadata)
    try:
        pretty_tabled_output(next(iter(index_tag)), index_tag[next(iter(index_tag))])  # output information
    except TypeError as errorlog:
        raise
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
#         return
if len(sys.argv) == 1:
    # file_path = "/Users/unauthorized/Downloads/models/image/auraflow.diffusers.1of2.fp16.safetensors"
    # file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"
    file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"

else:
    file_path = sys.argv[1]

if Path(file_path).is_dir() == True:
    path_data = os.listdir(file_path)
    print("\n\n\n\n")
    for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
        file_path = os.path.join(file_path, each_file)
        prepare_tags(file_path)

elif Path(file_path).exists:
    prepare_tags(file_path)

 # save_location)


# if __name__ == "__main__":
#     main()
