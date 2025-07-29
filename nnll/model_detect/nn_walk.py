# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# 從依賴關係中解析nn.module的實驗例程 experimental routines to parse nn.module out of dependencies
import os
import importlib
from importlib.util import find_spec
from inspect import isclass
from types import ModuleType
from typing import List, Tuple, Type


async def find_classes_in_package(pkg: ModuleType, base_class: Type) -> List[Tuple[str, Type]]:
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
        "diffusers.utils.import_utils",
        "diffusers.pipelines.consisid.consisid_utils",
        "diffusers.pipelines.stable_diffusion_safe",
        "diffusers.pipelines.skyreels_v2",  # demands ftfy dep
        "diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint_legacy",
        "diffusers.pipelines.stable_diffusion_k_diffusion",  # demands kdiffusion
        "diffusers.pipelines.deprecated",
        "diffusers.schedulers.scheduling_cosine_dpmsolver_multistep",  # needs torchsde
        "diffusers.schedulers.scheduling_dpmsolver_sde",
        "mlx_lm.models.olmo",  # demands al-olmo"
        "mlx_lm.test",
        "mlx_lm.evaluate",
        "torch.backends",  # demands coreml, onnx_script, etc
        "torch.utils.tensorboard",  # demands tensorboard
        "torch.testing",
    ]
    class_exclusions = ["pipeline_stable_diffusion_k_diffusion", "pipeline_onnx_stable_diffusion_inpaint", "CogView4PlusPipelineOutput", "OnnxStableDiffusionInpaintPipelineLegacy", "CogView3PlusPipelineOutput"]

    async def recurse(module_name: str):
        if module_name in visited:
            return
        visited.add(module_name)
        spec = find_spec(module_name)
        if not spec or not spec.origin:
            return
        if module_name not in exclude_list and "_tf" not in module_name and not any([segment.startswith("_") for segment in module_name.split(".")]):
            module = importlib.import_module(module_name)
            for name in dir(module):
                if name not in class_exclusions and "SkyReelsV2" not in name and not list([exclusion for exclusion in class_exclusions if exclusion in name]):
                    obj = getattr(module, name)
                    if isclass(obj) and issubclass(obj, base_class):
                        found.append((str(obj).replace("class ", ""), name))
            if spec.submodule_search_locations:
                path = spec.submodule_search_locations[0]
                for entry in os.listdir(path):
                    if entry.startswith("__") or entry in {"tests", "assets"}:
                        continue
                    full_path = os.path.join(path, entry)
                    if os.path.isdir(full_path):
                        await recurse(f"{module_name}.{entry}")
                    elif entry.endswith(".py"):
                        mod_name = entry[:-3]
                        if mod_name != "watermarking" and mod_name != "_VF":
                            await recurse(f"{module_name}.{mod_name}")

    await recurse(pkg.__name__)
    return found


async def find_modules():
    from importlib import import_module
    from nnll.metadata.json_io import write_json_file

    module_args = {
        "torch": "torch.nn",
    }
    results = []

    for dep, module in module_args.items():
        dep_obj = import_module(dep)
        module_obj = import_module(module).Module
        results.extend(await find_classes_in_package(dep_obj, module_obj))
    data = {name: cls for cls, name in results}
    write_json_file(".", "nn_sources.json", data)
    print(f"Wrote {len(results)} lines.")


async def find_nn():
    from importlib import import_module
    from nnll.metadata.json_io import write_json_file

    package_args = {
        "mlx_audio": "mlx.nn",
        "mlx_lm": "mlx.nn",
        "mflux": "mlx.nn",
        "transformers": "torch.nn",
        "diffusers": "torch.nn",
    }
    results = []
    for pkg, module in package_args.items():
        pkg_obj = import_module(pkg)
        module_obj = import_module(module).Module
        results.extend(await find_classes_in_package(pkg_obj, module_obj))
    data = {cls: name for cls, name in results}
    write_json_file(".", "nn_modules.json", data)
    print(f"Wrote {len(results)} lines.")


async def order_nn_modules():
    from nnll.metadata.json_io import read_json_file
    from nnll.metadata.json_io import write_json_file
    from collections import defaultdict
    import re

    key_sort = defaultdict(set)
    nn_sources: dict[str, str] = read_json_file("nn_sources.json")
    pkg_modules: dict[str, str] = read_json_file("nn_modules.json")
    for module, name in pkg_modules.items():
        model_name = None
        pattern = r"(\b(models.|mflux\.community.|pipelines.|generation.candidate_generator.|time_series_utils.|distributed.fsdp.|activations.|quantizers.base|transformers.loss.|transformers.integrations.|mlx_lm.dwq|nn.modules.loss)\b)"
        match = re.search(pattern, module)
        if match:
            match_segment = match.group()
            model_name = module.split(match_segment)[1].partition(".")[0]
            for nn_type in nn_sources.keys():
                if nn_type.lower() in name.lower():
                    key_sort[model_name].add(nn_type.lower())
        else:
            print(module)
    key_set = key_sort.copy()
    for i in key_set:
        key_sort[i] = list(key_set[i])
    write_json_file(".", "nn_order.json", key_sort)
    print(f"Wrote {len(key_sort)} lines.")


if __name__ == "__main__":
    import asyncio

    # asyncio.run(find_modules())
    # asyncio.run(find_nn())
    asyncio.run(order_nn_modules())
