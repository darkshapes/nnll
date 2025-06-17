# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Callable, Generator, Optional, Tuple, List, Dict

from gguf import Union
from nnll.monitor.file import nfo, dbug
from importlib import import_module


def scrape_docs(doc_string: str) -> Tuple[str,]:
    """Eat the 🤗Diffusers docstrings as a treat, leaving any tasty repo and class morsels neatly arranged as a dictionary.\n
    Nom.
    :param doc_string: String literal from library describing the class
    :return: A yummy dictionary of relevant class and repo strings
    """

    pipe_prefix = [">>> adapter = ", ">>> pipe_prior = ", ">>> pipe = ", ">>> pipeline = ", ">>> blip_diffusion_pipe = ", ">>> gen_pipe = ", ">>> prior_pipe = "]
    repo_prefixes = ["repo_id", "model_ckpt", "model_id_or_path", "model_id", "repo"]
    class_method = [".from_pretrained(", ".from_single_file("]
    staged_class_method = ".from_pretrain("
    staged = None
    staged_class = None
    staged_repo = None
    joined_docstring = " ".join(doc_string.splitlines())

    for prefix in pipe_prefix:
        pipe_doc = joined_docstring.partition(prefix)[2]  # get the string segment that follows pipe assignment
        if prefix == pipe_prefix[-2]:  # continue until loop end [exhaust last two items in list above]
            staged = pipe_doc
        elif pipe_doc and not staged:
            break
    for method_name in class_method:
        if method_name in pipe_doc:
            pipe_class = pipe_doc.partition(method_name)[0]  # get the segment preceding the class' method call
            repo_path = pipe_doc.partition(method_name)  # break segment at method
            repo_path = repo_path[2].partition(")")[0]  # segment after is either a repo path or a reference to it, capture the part before the parenthesis
            repo_path = repo_path.replace("...", "").strip()  # remove any ellipsis and empty space
            repo_path = repo_path.partition('",')[0]  # partition at commas, repo is always the first argument
            repo_path = repo_path.strip('"')  # strip remaining quotes
            # * the star below could go here?
            if staged:
                staged_class = staged.partition(staged_class_method)[0]  # repeat with any secondary stages
                staged_repo = staged.partition(staged_class_method)
                staged_repo = staged_repo[2].partition(")")[0]
                staged_repo = staged_repo.replace("...", "").strip()
                staged_repo = staged_repo.partition('",')[0]
                staged_repo = staged_repo.strip('"')
            break
        else:
            continue
    for prefix in repo_prefixes:  # * this could move up
        if prefix in repo_path and not staged:  # if  don't have the repo path, but only a reference
            repo_variable = f"{prefix} = "  # find the variable assignment
            repo_path = next(line.partition(repo_variable)[2].split('",')[0] for line in doc_string.splitlines() if repo_variable in line).strip('"')
            break
    return pipe_class, repo_path, staged_class, staged_repo


def cut_docs() -> Generator:
    """Draw down docstrings from 🤗Diffusers library, minimizing internet requests\n
    :return: Docstrings for common diffusers models
    """
    import pkgutil

    # from importlib import import_module
    import diffusers.pipelines

    non_standard = {
        "cogvideo": "cogvideox",
        "cogview3": "cogview3plus",
        "deepfloyd_if": "if",
        "cosmos": "cosmos2_text2image",  # search folder for all files containing 'EXAMPLE DOC STRING'
        "visualcloze": "visualcloze_generation",
    }

    exclusion_list = [  # task specific, adapter, or no doc string
        # these will be handled eventually
        "animatediff",  # adapter
        "controlnet",
        "controlnet_hunyuandit",  #: "hunyuandit_controlnet",
        "controlnet_xs",
        "controlnetxs",
        "controlnet_hunyuandit",
        "controlnet_sd3",
        "pag",  #
        "stable_diffusion_3_controlnet",
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
        "stable_diffusion_safe",  # impossibru
        "text_to_video_synthesis",
        "unidiffuser",
    ]

    for _, name, is_pkg in pkgutil.iter_modules(diffusers.pipelines.__path__):
        if is_pkg and name not in exclusion_list:
            if name in non_standard:
                file_specific = non_standard[name]
            else:
                file_specific = name
            file_name = f"pipeline_{file_specific}"
            try:
                pipe_file = import_module(f"diffusers.pipelines.{name}.{file_name}")
            except ModuleNotFoundError as error_log:
                nfo(f"Module Not Found for {name}")
                dbug(error_log)
                pipe_file = None
            try:
                if pipe_file:
                    yield pipe_file.EXAMPLE_DOC_STRING
            except AttributeError as error_log:
                nfo(f"Doc String Not Found for {name}")
                dbug(error_log)
                # print(sub_classes)


def root_class(init_module: Union[Callable, str], library: Optional[str] = None) -> Dict[str, List[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param init_module: Origin pipeline
    :return: Dictionary of
    """
    import importlib
    import inspect

    if library and isinstance(init_module, str):
        base_library = importlib.import_module(library)
        init_module = getattr(base_library, init_module)
    signature = inspect.signature(init_module.__init__)
    class_names = {}
    for folder, param in signature.parameters.items():
        if folder != "self":
            sub_module = str(param.annotation).split("'")
            if len(sub_module) > 1 and sub_module[1] not in ["bool", "int", "float", "complex", "str", "list", "tuple", "dict", "set"]:
                class_names.setdefault(folder, sub_module[1].split("."))
    return class_names


def get_code_names(class_name: Optional[str] = None, diffusers: bool = False) -> Union[List[str], str]:
    """Reveal code names for class names from Diffusers or Transformers\n
    :param class_name: To return only one class, defaults to None
    :param diffusers: To search in Diffusers, `False` for Transformers, defaults to False
    :return: A list of all code names, or the one corresponding to the provided class
    """
    if diffusers:
        if class_name:
            from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING

            return next(iter(k for k, v in AUTO_TEXT2IMAGE_PIPELINES_MAPPING.items() if class_name in v), "")
        from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING as MAPPING_NAMES
    else:
        if class_name:
            from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

            return next(iter(k for k, v in MODEL_MAPPING_NAMES.items() if class_name in v), "")
        from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES as MAPPING_NAMES
    return list(MAPPING_NAMES)


def show_tasks(class_name: Optional[str] = None, code_name: Optional[str] = None) -> List[str]:
    """Return Diffusers/Transformers task pipes based on package-specific query\n
    :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
    :param code_name: To find task pipes from a Transformers class pipe, defaults to None
    :return: _description_
    """
    if class_name:
        from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

        alt_tasks = []
        for task_map in SUPPORTED_TASKS_MAPPINGS:
            task_class = _get_task_class(task_map, class_name, False)
            if task_class:
                alt_tasks.append(task_class.__name__)
    elif code_name:
        from transformers.utils.fx import _generate_supported_model_class_names

        alt_tasks = _generate_supported_model_class_names(code_name)
    return alt_tasks


def collate_transformer_data():
    transformer_data = {}
    exclude_list = ["DistilBertModel", "SeamlessM4TModel", "SeamlessM4Tv2Model"]
    for i in get_code_names():
        task_pipe = next(iter(show_tasks(code_name=i)))
        if isinstance(task_pipe, tuple):
            task_pipe = task_pipe[0]
        if task_pipe not in exclude_list:
            model_class = getattr(__import__("transformers"), task_pipe)
            model_data = root_class(model_class)
            if model_data and "inspect" not in model_data["config"] and "deprecated" not in model_data["config"]:
                transformer_data.setdefault(model_class.__name__, list(root_class(model_data["config"][-1], library="transformers")))
    return transformer_data


def pull_weight_map(repo_id: str, arch: str) -> Dict[str, str]:
    from nnll.download.hub_cache import download_hub_file

    model_file = download_hub_file(repo_id=f"{repo_id}/tree/main/{arch}", source="huggingface", file_name="diffusion_pytorch_model.safetensors.index.json", local_dir=".tmp")
