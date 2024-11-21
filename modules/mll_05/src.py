
"""
Practical analysis and comparison of two methods for loading GGUF files.
ungguf   - llama cpp python
unggufed - gguf (from py lib of llama cpp)
`file_name` path to gguf file to load
`id_values` data inferred from the model and its header
19/11/24 Verdict : use llama cpp
"""

import os
import struct
from collections import defaultdict
from llama_cpp import Llama
from gguf import GGUFReader  # smaller import?


def ungguf(file_name: str, id_values: dict):

    id_values["file_size"] = os.path.getsize(file_name)  # how big will be important for memory management
    file_data = defaultdict(dict)
    try:
        with open(file_name, "rb") as file:
            magic = file.read(4)
            if magic != b"GGUF":
                print(f"Invalid GGUF magic number in '{file_name}'")  # uh uh uh, you didn't say the magic word
                return
            version = struct.unpack("<I", file.read(4))[0]
            if version < 2:
                print(f"Unsupported GGUF version {version} in '{file_name}'")
                return
    except ValueError as error_log:
        print(error_log)  # the aforementioned failing
    else:
        try:
            parser = Llama(model_path=file_name, vocab_only=True, verbose=False)  # fails image quants, but dramatically faster vs ggufreader
        except:
            pass
        else:
            arch = parser.metadata.get("general.architecture")  # with gguf we can directly request the model name but it isnt always written in full
            name = parser.metadata.get("general.name")  # sometimes this field is better than arch
            id_values["name"] = name if name is not None else arch
            id_values["dtype"] = parser.scores.dtype.name  # outputs as full name eg: 'float32 rather than f32'
            return id_values


def unggufed(file_name: str, id_values: dict):
    try:  # method using gguf library, better for LDM conversions
        reader = GGUFReader(file_name, "r")  # obsolete in numpy 2, also slower
    except ValueError as error_log:
        print(error_log)  # >:V
    else:
        id_values["dtype"] = reader.data.dtype.name  # get dtype from metadata
        print(reader.fields.get("general.name"))
        arch = reader.fields["general.architecture"]  # model type category, usually prevents the need  toblock scan for llms
        id_values["name"] = str(bytes(arch.parts[arch.data[0]]), encoding="utf-8")  # retrieve model name from the dict data
        if len(arch.types) > 1:
            id_values["name"] = arch.types  # if we get a result, save it
        for tensor in reader.tensors:
            id_values[str(tensor.name)] = {"shape": str(tensor.shape), "dtype": str(tensor.tensor_type.name)}  # create dict similar to safetensors/pt results
        return id_values


# id_values_00 = defaultdict(dict)
# id_values_01 = defaultdict(dict)

# file_name = "/Users/unauthorized/Downloads/models/text/suzume-llama-3-8B-multilingual-orpo-borda-top25.IQ4_NL.gguf"
# virtual_data_00 = ungguf(file_name, id_values_00)
# virtual_data_01 = unggufed(file_name, id_values_01)
# print(virtual_data_00["blk.0.attn_norm.weight"].get("shape"))  # preferred
# print(virtual_data_01["blk.0.attn_norm.weight"].shape)
