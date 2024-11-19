import struct
import json
from safetensors.torch import load_file


def __unsafetensors(file_name: str):
    with open(file_name, 'rb') as file:
        try:
            first_8_bytes = file.read(8)
            length_of_header = struct.unpack('<Q', first_8_bytes)[0]
            header_bytes = file.read(length_of_header)
            header = json.loads(header_bytes.decode('utf-8'))
            # we want to remove this metadata so its not counted as tensors
            if header.get("__metadata__", 0) != 0:
                # it is usually empty on safetensors ._.
                header.pop("__metadata__")
            return header
        except Exception as error_log:  # couldn't open file
            print(error_log)


def __untorchtensors(file_name: str):
    try:
        return load_file(file_name)
    except Exception as error_log:  # couldn't open file
        print(error_log)


file_name = "~/Downloads/models/lora/Hyper-FLUX.1-dev-8steps-lora.safetensors"
virtual_data_00 = __unsafetensors(file_name)
virtual_data_01 = __untorchtensors(file_name)
print(virtual_data_00["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"].get("shape"))  # preferred
print(virtual_data_01["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"].shape)
