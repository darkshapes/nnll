
import struct
import json


def load_safetensors_metadata(file_path: str) -> dict:
    """
    Collect metadata from a safetensors file header\n
    :param file_path: `str` the full path to the file being opened
    :return: `dict` the key value pair structure found in the file
    """
    with open(file_path, 'rb') as file:
        first_8_bytes = file.read(8)
        length_of_header = struct.unpack('<Q', first_8_bytes)[0]
        header_bytes = file.read(length_of_header)
        header = json.loads(header_bytes.decode('utf-8'))
        # we want to remove this metadata so its not counted as tensors
        if header.get("__metadata__", 0) != 0:
            # it is usually empty on safetensors ._.
            header.pop("__metadata__")
        return header


file_name = "~/Downloads/models/lora/Hyper-FLUX.1-dev-8steps-lora.safetensors"
virtual_data_00 = load_safetensors_metadata(file_name)
print(virtual_data_00["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"].get("shape"))  # preferred
