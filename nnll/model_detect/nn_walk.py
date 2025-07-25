import os
import importlib
from importlib.util import find_spec
from inspect import isclass
from types import ModuleType
from typing import List, Tuple, Type


def find_classes_in_package(pkg: ModuleType, base_class: Type) -> List[Tuple[str, Type]]:
    visited = set()
    found = []
    exclude_list = [
        "watermarking",
        "transformers.utils.sentencepiece_model_pb2",
        "transformers.kernels.falcon_mamba",
        "selective_scan_with_ln_interface",
        "transformers.utils.notebook",
        "transformers.models.biogpt.modular_biogpt",
        "transformers.models.wavlm.modular_wavlm",
        "transformers.models.gemma3.modular_gemma3",
        "transformers.models.data2vec.modular_data2vec_audio",
        "transformers.models.gemma3n.modular_gemma3n",
        "transformers.models.pop2piano.tokenization_pop2piano",
        "transformers.models.dinov2_with_registers.modular_dinov2_with_registers",
        "transformers.models.kyutai_speech_to_text.modular_kyutai_speech_to_text",
        "transformers.integrations",
        "transformers.keras_callbacks",
        "transformers.tf_utils",
        "transformers.generation.tf_logits_process",
        "transformers.generation.tf_utils",
    ]

    def recurse(module_name: str):
        if module_name in visited:
            return
        visited.add(module_name)
        print(visited)
        spec = find_spec(module_name)
        if not spec or not spec.origin:
            return
        if module_name not in exclude_list and "_tf" not in module_name and "__" not in module_name:
            print(module_name)
            module = importlib.import_module(module_name)
            for name in dir(module):
                obj = getattr(module, name)
                if isclass(obj) and issubclass(obj, base_class):
                    found.append((name, obj))
            if spec.submodule_search_locations:
                path = spec.submodule_search_locations[0]
                for entry in os.listdir(path):
                    if entry.startswith("__") or entry in {"tests", "assets"}:
                        continue
                    full_path = os.path.join(path, entry)
                    if os.path.isdir(full_path):
                        recurse(f"{module_name}.{entry}")
                    elif entry.endswith(".py"):
                        mod_name = entry[:-3]
                        if mod_name != "watermarking":
                            recurse(f"{module_name}.{mod_name}")

    recurse(pkg.__name__)
    return found


def find_nn():
    import mlx_audio
    import mlx.nn as nn
    from nnll.metadata.json_io import write_json_file

    results = find_classes_in_package(mlx_audio, nn.Module)

    import transformers
    import torch.nn as nn

    results = find_classes_in_package(transformers, nn.Module)
    data = {name: str(cls) for name, cls in results}
    write_json_file(".", "nn_modules.json", data)
    print(f"Wrote {len(results)} lines.")


if __name__ == "__main__":
    find_nn()
