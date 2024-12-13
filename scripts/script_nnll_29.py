
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.path[0]))))
from modules.nnll_04.src import load_safetensors_metadata
from modules.nnll_05.src import load_gguf_metadata
from modules.nnll_28.src import load_pickletensor_metadata
from modules.nnll_07.src import Domain, Architecture, Component
from modules.nnll_27.src import pretty_tabled_output
from modules.nnll_29.src import BlockScanner
from modules.nnll_30.src import read_json_file, write_json_file


def run(file_path: str) -> None:
    method_map = {
        ".safetensors": load_safetensors_metadata,
        ".sft": load_safetensors_metadata,
        ".gguf": load_gguf_metadata,
        ".pt": load_pickletensor_metadata,
        ".pth": load_pickletensor_metadata,
        ".ckpt": load_pickletensor_metadata
    }
    FILTER = read_json_file("modules/nnll_29/filter.json")

    # Process file by method indicated by extension, usually struct unpacking, except for pt files which are memmap
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_extension = Path(file_path).suffix.lower()  # extensions are metadata
    if file_extension == "" or file_extension is None or file_extension not in method_map:
        return

    model_header = method_map[file_extension](file_path)

    try:  # Be sure theres something in model_header
        next(iter(model_header))
    except TypeError as errorlog:
        raise TypeError(errorlog)
    else:  # Process and output metadata
        tensor_count = { "tensors": len(model_header) }
        block_scan = BlockScanner()
        file_metadata = block_scan.filter_metadata(FILTER, model_header, tensor_count)
        for element in file_metadata:
            if isinstance(file_metadata[element], list):
                file_metadata[element] = ' '.join(file_metadata[element])
        domain_ml = Domain("ml")  # create the domain only when we know its a model
        arch_found = Architecture(file_metadata.get("model"))
        comp_inside = Component(file_metadata["category"], disk_size=file_size, disk_path=file_path, layer_type=file_metadata["layer_type"], file_name=file_name)
        arch_found.add_component(comp_inside.model_type, comp_inside)
        domain_ml.add_architecture(arch_found.architecture, arch_found)
        model_index_dict = domain_ml.to_dict()
        try:
            pretty_tabled_output(next(iter(model_index_dict)), model_index_dict[next(iter(model_index_dict))])  # output information
        except TypeError as errorlog:
            raise TypeError(errorlog)

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
    file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"
    file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"
else:
    file_path = sys.argv[1]

if Path(file_path).is_dir() == True:
    path_data = os.listdir(file_path)
    print("\n\n\n\n")
    for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
        file_path = os.path.join(file_path, each_file)
        run(file_path)  # save_location)
elif Path(file_path).exists:
    run(file_path)  # save_location)


# if __name__ == "__main__":
#     main()

# # file_path = "/Users/unauthorized/Downloads/models/image/hunyuandit1.2.safetensors"
# file_path = "/Users/unauthorized/Downloads/models/HunyuanDiT-v1.2-Diffusers/transformer/diffusion_pytorch_model.safetensors"
