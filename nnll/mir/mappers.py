# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Dict, List, Generator, Iterator, Tuple
import pkgutil
import diffusers.pipelines
import sys
from nnll.metadata.helpers import make_callable
from nnll.tensor_pipe.deconstructors import root_class
from nnll.tensor_pipe.parenting import show_tasks_for

nfo = sys.stderr.write


def stock_llm_data() -> Dict[str, List[str]]:
    """Eat the 🤗Transformers classes as a treat, leaving any tasty subclass class morsels neatly arranged as a dictionary.\n
    Nom.
    :return: _description_"""

    transformer_data = {}
    exclude_list = ["DistilBertModel", "SeamlessM4TModel", "SeamlessM4Tv2Model"]
    import os

    import transformers
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    model_names = list(dict(MODEL_MAPPING_NAMES).keys())
    folder_data = {*model_names}
    models_folder = os.path.join(os.path.dirname(transformers.__file__), "models")
    folder_data = folder_data.union(os.listdir(models_folder))

    for code_name in folder_data:
        if code_name and "__" not in code_name:
            tasks = show_tasks_for(code_name=code_name)
            if tasks:
                task_pipe = next(iter(tasks))
                if isinstance(task_pipe, tuple):
                    task_pipe = task_pipe[0]
                if task_pipe not in exclude_list:
                    model_class = getattr(__import__("transformers"), task_pipe)
                    model_data = root_class(model_class)
                    if model_data and "inspect" not in model_data["config"] and "deprecated" not in model_data["config"]:
                        transformer_data.setdefault(model_class, model_data)
    return transformer_data


def process_with_folder_path(pkg_name: str, folder_path: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package folder paths to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module
    import os

    file_names = list(getattr(folder_path, "_import_structure").keys())
    module_path = os.path.dirname(import_module("diffusers.pipelines").__file__)
    for file_name in file_names:
        if file_name == "pipeline_stable_diffusion_xl_inpaint":
            continue
        try:
            pkg_path = f"diffusers.pipelines.{pkg_name}.{file_name}"
            path_exists = os.path.exists(os.path.join(module_path, pkg_name, file_name + ".py"))
            if path_exists:
                pipe_file = make_callable(file_name, pkg_path)
        except ModuleNotFoundError:
            nfo(f"Module Not Found for {pkg_name}")
            pipe_file = None

        try:
            if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
                yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
            else:
                if path_exists:
                    pipe_file = import_module(pkg_path)
        except (ModuleNotFoundError, AttributeError):
            nfo(f"Doc String Not Found for {pipe_file} {pkg_name}")


def process_with_file_name(pkg_name: str, file_specific: bool) -> Iterator[Tuple[str, str, str]]:
    """Processes package using file name to yield example doc strings if available.\n
    :param pkg_name: The name of the package under diffusers.pipelines.
    :param file_specific: A flag indicating whether processing is specific to certain files.
    :yield: A tuple containing (pkg_name, file_name, EXAMPLE_DOC_STRING) if found.
    """
    from importlib import import_module

    file_name = f"pipeline_{file_specific}"
    try:
        pkg_path = f"diffusers.pipelines.{pkg_name}"
        pipe_file = make_callable(file_name, pkg_path)
    except ModuleNotFoundError:
        nfo(f"Module Not Found for {pkg_name}")
        pipe_file = None
    try:
        if pipe_file and hasattr(pipe_file, "EXAMPLE_DOC_STRING"):
            yield (pkg_name, file_name, pipe_file.EXAMPLE_DOC_STRING)
        else:
            pipe_file = import_module(pkg_path)
    except AttributeError:
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
        "diffusionpipeline",  #
        "pag",  # not model based
        "stable_diffusion_attend_and_excite",
        "stable_diffusion_sag",  #
        "t2i_adapter",
        "ledits_pp",  # "leditspp_stable_diffusion",
        "latent_consistency_models",  # "latent_consistency_text2img",
        "unclip",
        # these are uncommon afaik
        "dance_diffusion",  # no doc_string
        "dit",
        "ddim",
        "ddpm",
        "deprecated",
        "latent_diffusion",  # no doc_string
        "marigold",  # specific processing routines
        "omnigen",  # tries to import torchvision
        "paint_by_example",  # no docstring
        "pia",  # lora adapter
        "semantic_stable_diffusion",  # no_docstring
        "stable_diffusion_diffedit",
        "stable_diffusion_k_diffusion",  # tries to import k_diffusion
        "stable_diffusion_panorama",
        "stable_diffusion_safe",  # impossible
        "text_to_video_synthesis",
        "unidiffuser",
    ]

    for _, pkg_name, is_pkg in pkgutil.iter_modules(diffusers.pipelines.__path__):
        if is_pkg and pkg_name not in exclusion_list:
            file_specific = non_standard.get(pkg_name, pkg_name)
            folder_name = getattr(diffusers.pipelines, pkg_name)
            if folder_name:
                if hasattr(folder_name, "_import_structure"):
                    yield from process_with_folder_path(pkg_name, folder_name)
                else:
                    yield from process_with_file_name(pkg_name, file_specific)
            else:
                continue
