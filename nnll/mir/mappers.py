# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pkgutil
from typing import Dict, Generator, Iterator, List, Tuple

import diffusers

from nnll.metadata.helpers import make_callable
from nnll.tensor_pipe.deconstructors import root_class

nfo = print


def stock_llm_data() -> Dict[str, List[str]]:
    """Eat the 🤗Transformers classes as a treat, leaving any tasty subclass class morsels neatly arranged as a dictionary.\n
    Nom.
    :return: Tasty mapping of subclasses to their class references"""

    transformer_data = {}
    exclude_list = [
        "DecisionTransformerModel",
        "DistilBertModel",
        "GraphormerModel",
        "GPTBigCodeModel",
        "TimmBackbone",
        "PerceptionEncoder",
        "SeamlessM4Tv2Model",
        "SeamlessM4TModel",
        "VisionTextDualEncoderModel",
    ]
    second_exclude_list = [
        "vision-text-dual-encoder",
        "vision_text_dual_encoder",
        "gpt_bigcode",
        "data2vec",
        "bert_japanese",
        "cpm",
        "dab_detr",
        "decision_transformer",
        "timm_backbone",
    ]  # there just isnt a repo in this one
    import os

    import transformers
    from transformers.models.auto.modeling_auto import CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES

    from nnll.model_detect.tasks import AutoPkg

    model_data = None
    task_pipe = None
    model_names = list(dict(MODEL_MAPPING_NAMES).keys())
    folder_data = {*model_names}
    models_folder = os.path.join(os.path.dirname(transformers.__file__), "models")
    folder_data = folder_data.union(os.listdir(models_folder))
    for code_name in folder_data:
        model_class = None
        if code_name and "__" not in code_name:
            tasks = AutoPkg.show_transformers_tasks(code_name=code_name)
            if tasks:
                task_pipe = next(iter(tasks))
                if isinstance(task_pipe, tuple):
                    task_pipe = task_pipe[0]
                if task_pipe not in exclude_list:
                    model_class = getattr(__import__("transformers"), task_pipe)  # this is done to get the path to the config
                    model_data = root_class(model_class)
                    if model_data and ("inspect" not in model_data["config"]) and ("deprecated" not in list(model_data["config"])):
                        transformer_data.setdefault(model_class, model_data)
                    else:
                        model_data = None

            if not model_data and code_name not in second_exclude_list:  # second attempt
                if code_name == "donut":
                    code_name = "donut-swin"
                if not task_pipe and code_name and MODEL_MAPPING_NAMES.get(code_name.replace("_", "-")):
                    model_class = getattr(__import__("transformers"), MODEL_MAPPING_NAMES[code_name.replace("_", "-")], None)
                elif task_pipe:
                    model_class = getattr(__import__("transformers"), task_pipe)
                config_class = CONFIG_MAPPING_NAMES.get(code_name.replace("_", "-"))
                if not config_class:
                    config_class = CONFIG_MAPPING_NAMES.get(code_name.replace("-", "_"))
                if config_class:
                    config_class_obj = getattr(__import__("transformers"), config_class)
                    model_data = {"config": str(config_class_obj.__module__ + "." + config_class_obj.__name__).split(".")}
                    if model_data and ("inspect" not in model_data) and ("deprecated" not in model_data) and model_class:
                        transformer_data.setdefault(model_class, model_data)
    return transformer_data


def pkg_path_to_docstring(pkg_name: str, folder_path: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package folder paths to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    import os
    from importlib import import_module

    file_names = list(getattr(folder_path, "_import_structure").keys())
    module_path = os.path.dirname(import_module("diffusers.pipelines").__file__)
    for file_name in file_names:
        if file_name == "pipeline_stable_diffusion_xl_inpaint":
            continue
        try:
            pkg_path = f"diffusers.pipelines.{str(pkg_name)}.{file_name}"
            print(pkg_path)
            path_exists = os.path.exists(os.path.join(module_path, pkg_name, file_name + ".py"))
            if path_exists:
                pipe_file = make_callable(file_name, pkg_path)
        except ModuleNotFoundError:
            if pkg_name != "skyreels_v2":
                nfo(f"Module Not Found for {pkg_name}")
            pipe_file = None

        try:
            if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
                yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
            else:
                if path_exists:
                    pipe_file = import_module(pkg_path)
        except (ModuleNotFoundError, AttributeError):
            if pkg_name != "skyreels_v2":
                nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


def file_name_to_docstring(pkg_name: str, file_specific: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package using file name to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module

    file_name = f"pipeline_{file_specific}"
    try:
        pkg_path = f"diffusers.pipelines.{str(pkg_name)}"
        pipe_file = make_callable(file_name, pkg_path)
    except ModuleNotFoundError:
        if pkg_name != "skyreels_v2":
            nfo(f"Module Not Found for {pkg_name}")
        pipe_file = None
    try:
        if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
            yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
        else:
            pipe_file = import_module(pkg_path)

    except AttributeError:
        if pkg_name != "skyreels_v2":
            nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


# Refactored main loop


def cut_docs() -> Generator:
    """Draw down docstrings from 🤗Diffusers library, minimizing internet requests\n
    :return: Docstrings for common diffusers models"""

    non_standard = {
        "cogvideo": "cogvideox",
        "cogview3": "cogview3plus",
        "deepfloyd_if": "if",
        "cosmos": "cosmos2_text2image",  # search folder for all files containing 'EXAMPLE DOC STRING'
        "visualcloze": "visualcloze_generation",
    }

    exclusion_list = [  # no doc string or other issues. all can be be gathered by other means
        "autopipeline",  #
        "dance_diffusion",  # no doc_string
        "ddim",
        "ddpm",
        "deprecated",
        "diffusionpipeline",  #
        "dit",
        "latent_consistency_models",  # "latent_consistency_text2img",
        "latent_diffusion",  # no doc_string
        "ledits_pp",  # "leditspp_stable_diffusion",
        "marigold",  # specific processing routines
        "omnigen",  # tries to import torchvision
        "pag",  # not model based
        "paint_by_example",  # no docstring
        "pia",  # lora adapter
        "semantic_stable_diffusion",  # no_docstring
        "stable_diffusion_attend_and_excite",
        "stable_diffusion_diffedit",
        "stable_diffusion_k_diffusion",  # tries to import k_diffusion
        "stable_diffusion_panorama",
        "stable_diffusion_safe",  # impossible
        "stable_diffusion_sag",  #
        "t2i_adapter",
        "text_to_video_synthesis",
        "unclip",
        "unidiffuser",
        # these are uncommon afaik
    ]

    for _, pkg_name, is_pkg in pkgutil.iter_modules(diffusers.pipelines.__path__):
        if is_pkg and pkg_name not in exclusion_list:
            file_specific = non_standard.get(pkg_name, pkg_name)
            folder_name = getattr(diffusers.pipelines, str(pkg_name))
            if folder_name:
                if hasattr(folder_name, "_import_structure"):
                    yield from pkg_path_to_docstring(pkg_name, folder_name)
                else:
                    yield from file_name_to_docstring(pkg_name, file_specific)
            else:
                continue
