# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

from typing import Callable, Generator, Tuple, List, Dict
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
    pretrained_prefix = [".from_pretrained(", ".from_single_file("]
    staged_prefix = ".from_pretrain("
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
    for pretrained in pretrained_prefix:
        pipe_class = pipe_doc.partition(pretrained)[0]  # get the segment preceding the class' method call
        repo_path = pipe_doc.partition(pretrained)  # break segment at method
        repo_path = repo_path[2].partition(")")[0]  # segment after is either a repo path or a reference to it, capture the part before the parenthesis
        repo_path = repo_path.replace("...", "").strip()  # remove any ellipsis and empty space
        repo_path = repo_path.partition('",')[0]  # partition at commas, repo is always the first argument
        repo_path = repo_path.strip('"')  # strip remaining quotes
        # * the star below could go here?
        if staged:
            staged_class = staged.partition(staged_prefix)[0]  # repeat with any secondary stages
            staged_repo = staged.partition(staged_prefix)
            staged_repo = staged_repo[2].partition(")")[0]
            staged_repo = staged_repo.replace("...", "").strip()
            staged_repo = staged_repo.partition('",')[0]
            staged_repo = staged_repo.strip('"')
        break
    for prefix in repo_prefixes:  # * this could move up
        if prefix in repo_path and not staged:  # if  don't have the repo path, but only a reference
            prefix_assign = f"{prefix} = "  # find the variable assignment
            repo_path = next(line.partition(prefix_assign)[2].split('",')[0] for line in doc_string.splitlines() if prefix_assign in line).strip('"')
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


def _get_task_pipe(pkg_name: dict, i2i: bool = False) -> list[Callable]:
    """Convert normal diffusers pipe to a task-specific pipe\n
    :return: a list of Callable element to import
    """
    from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

    task = "IMAGE2IMAGE" if i2i else "INPAINT"
    task_pipe = _get_task_class(next(iter(x for x in SUPPORTED_TASKS_MAPPINGS if task in x)), pkg_name.get("diffusers"))
    return [task_pipe]


def root_class(init_module: Callable) -> Dict[str, List[str]]:
    """Pick apart a Diffusers or Transformers pipeline class and find its constituent parts\n
    :param init_module: Origin pipeline
    :return: Dictionary of
    """
    import inspect

    signature = inspect.signature(init_module.__init__)
    class_names = {}
    for folder, param in signature.parameters.items():
        if folder != "self":
            sub_module = str(param.annotation).split("'")
            if len(sub_module) > 1 and sub_module[1] not in ["bool", "int", "float", "complex", "str", "list", "tuple", "dict", "set"]:
                class_names.setdefault(folder, sub_module[1].split("."))
    return class_names


def pull_weight_map(repo_id: str, arch: str) -> Dict[str, str]:
    from nnll.download.hub_cache import download_hub_file

    model_file = download_hub_file(repo_id=f"{repo_id}/tree/main/{arch}", source="huggingface", file_name="diffusion_pytorch_model.safetensors.index.json", local_dir=".tmp")


# from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

# from typing import Callable, Dict, List
# # from mir.mir_maid import MIRDatabase

# from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS
# from transformers.models.auto.modeling_auto import (
#     MODEL_MAPPING_NAMES,
#     MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
#     MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
#     MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
# )


# _REAL_CHECKPOINT_FOR_DOC
# _CHECKPOINT_FOR_DOC
# _IMAGE_CLASS_CHECKPOINT


# EXAMPLE_DOC_STRING


# def list_transformers_models():
#     import pkgutil
#     import transformers.models

#     for _, name, is_pkg in pkgutil.iter_modules(transformers.models.__path__):
#         if is_pkg:
#             print(name)


# from transformers.utils.fx import _generate_supported_model_class_names

# short_name = next(iter(x for x in DIFFUSERS_DEFAULT_PIPELINE_PATHS if "refiner" in x))

# _REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS = [
#     "altclip",
#     "albert",
#     "bart",
#     "bert",
#     "bitnet",
#     "blenderbot",
#     "blenderbot-small",
#     "bloom",
#     "clip",
#     "convnext",
#     "deberta",
#     "deberta-v2",
#     "dinov2",
#     "distilbert",
#     "donut-swin",
#     "electra",
#     "gpt2",
#     "gpt_neo",
#     "gptj",
#     "hiera",
#     "hubert",
#     "ijepa",
#     "layoutlm",
#     "llama",
#     "cohere",
#     "lxmert",
#     "m2m_100",
#     "marian",
#     "mbart",
#     "megatron-bert",
#     "mistral",
#     "mixtral",
#     "mobilebert",
#     "mt5",
#     "nezha",
#     "opt",
#     "pegasus",
#     "plbart",
#     "qwen2",
#     "qwen2_moe",
#     "qwen3",
#     "qwen3_moe",
#     "resnet",
#     "roberta",
#     "segformer",
#     "speech_to_text",
#     "speech_to_text_2",
#     "swin",
#     "t5",
#     "trocr",
#     "vit",
#     "xglm",
#     "wav2vec2",
#     #    "xlnet",
# ]

# infer_diffusers_model_type


# def find_pipe(query: str):
#     from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING


# import ast
# import importlib
# from pathlib import Path
# import os

# package = importlib.import_module('diffusers.schedulers')
# file_path = Path(package.__file__).parent / '__init__.py'

# with open(os.path.abspath(file_path.resolve()), 'r') as file:
#     tree = ast.parse(file.read())

# class_imports = []

# for node in ast.walk(tree):
#     if isinstance(node, (ast.Import, ast.ImportFrom)):
#         for alias in node.names:
#             # Assuming class names are capitalized
#             if alias.name[0].isupper():
#                 class_imports.append(alias.name)

# print(class_imports)


# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
# import os
# , List, Union
# from nnll.monitor.file import debug_monitor, dbug, nfo

# from mir.constants import ChipType, PkgType
# from mir.mir_maid import MIRDatabase


# # turn repo into class based on pkg/chip
# PkgType.check_type("MFLUX") + ChipType.MPS
# PkgType.DIFFUSERS
# PkgType.BITSANDBYTES
# PkgType.TORCH

# check = {CueType.HUB}


# def filter_dependencies(model_path: str, mir_db: Callable = MIRDatabase()):
#     mir_tag = mir_db.find_path(field="repo", target=model_path)


#     if "".join(mir_tag).find("flux") > 0 and CueType.check_type("MFLUX") and "MPS" in ChipType._show_ready("mps"):
#         nfo("HAS MFLUX")
#     # pylint:disable=no-member
#     return mir_tag
#     # if CueType.HUB:


#     from huggingface_hub import hf_hub_download


#     repos = [
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         "stabilityai/stable-diffusion-xl-refiner-1.0",
#         "kwai-kolors/kolors-diffusers",
#         "stabilityai/stable-cascade",
#         "stabilityai/stable-cascade-prior",
#         "playgroundai/playground-v2.5-1024px-aesthetic",
#         "stabilityai/stable-diffusion-3.5-medium",
#         "adamo1139/stable-diffusion-3.5-medium-ungated",
#         "stabilityai/stable-diffusion-3.5-large",
#         "adamo1139/stable-diffusion-3.5-large-ungated",
#         "yandex/stable-diffusion-3.5-large-alchemist",
#         "yandex/stable-diffusion-3.5-medium-alchemisttensorart/stable-diffusion-3.5-medium-turbo",
#     ]
#     domain = "info"
#     kwargs = {"arch": "", "series": "", "comp": ""}
#     for repo in repos:
#         hf_hub_download(repo_id=repo, filename="model_index.json", local_dir=".tmp")


# def _get_module(self, import_pkg: dict[str, list[str]]) -> list[Callable]:
#     """Accept two lists of importable dependencies and modules\n
#     :param pkg_name: Main external dependencies
#     :param module_path: Sub-modules of the main dependency
#     :return: A list of callable statements
#     """
#     import importlib

#     import_modules = []
#     for name, module in import_pkg.items():
#         try:
#             pkg = importlib.import_module(name)
#         except (ImportError, AttributeError):
#             continue
#         else:
#             if not next(iter(module)):
#                 return
#             elif len(module) > 1:
#                 import_modules.append(getattr(pkg, ".".join(module)))
#             else:
#                 import_modules.append(getattr(pkg, next(iter(module))))
#     return import_modules


# def _load_pipe(self, pipe_class: str, repo: str, import_pkg: str, **kwargs) -> Callable:
#     if import_pkg in ["diffusers", "transformers", "parler-tts"]:
#         if os.path.isfile(repo):
#             pipe = pipe_class.from_single_file(repo, **kwargs)
#             return pipe
#         # from config
#         else:
#             pipe = pipe_class.from_pretrained(repo, **kwargs)
#             return pipe
#     elif import_pkg == "audiogen":
#         pipe_class = pipe_class.get_pretrained(repo, **kwargs)
#         return pipe
#     if pipe_class is None:
#         raise TypeError("Pipe should be Callable `class` object, not `None`")


# def add_lora(self, pipe: Callable, lora_repo: str, init_kwargs: dict, scheduler_data=None, scheduler_kwargs=None):
#     if scheduler_data:
#         import_pkg = scheduler_data["dep_pkg"]
#         scheduler_class = self._get_module(import_pkg)
#         pipe.scheduler = scheduler_class[0]({**scheduler_kwargs})
#         nfo(f"mid sched {scheduler_data}, {scheduler_class}")
#     # nfo(f"status mid-lora: {lora}, {arch_data}, {pipe}, {scheduler}, ")

#     fuse = 0
#     if init_kwargs is not None:
#         fuse = init_kwargs.get("fuse", 0)
#     pipe.load_lora_weights(lora_repo, adapter_name=os.path.basename(lora_repo))
#     if fuse:
#         pipe.fuse_lora(adapter_name=os.path.basename(lora_repo), lora_scale=fuse)
#         pipe.unload_lora_weights()
#     return pipe


# @debug_monitor
# @pipe_call
# def create_pipeline(self, arch_data: Union[List[str], str], init_modules: dict, *args, granular: bool = False, **kwargs):
#     """
#     Build an inference pipe based on model type\n
#     :param arch_data: Identifier of model architecture
#     :param init_modules: Parameters for initialiing the pipeline
#     :return: `tuple` constructed pipe, model/repo name `str`, and a `dict` of default settings
#     """

#     if isinstance(arch_data, str):
#         "".join(arch_data)
#     # if series.split(".")[1] == "lora":
#     #     scheduler = arch_data.get("solver", 0)
#     # granular should be set by spec check
#     import_pkg = init_modules["dep_pkg"]  # a dictionary of dependencies
#     # handle alternate external dependencies that are not installed
#     # if arch_data.get("dep_alt", 0):
#     #     import_pkg.update(arch_data["dep_alt"])
#     pipe_classes = self._get_module(import_pkg)  # now a list of classes
#     if granular and ("diffusers" in import_pkg or "transformers" in import_pkg):
#         # break down pipeline for low-spec machines
#         sub_cls_locs = self._get_sub_cls_locs(pipe_classes[-1])
#         # for folder, package in sub_cls_locs:
#         # self._get_module(package[:1], package[-1:])
#         # self.construct.find_path()

#     repo_paths = arch_data.get("repo")
#     init_kwargs = arch_data.get("init_kwargs", {})
#     init_kwargs.update(kwargs)  # add user kwargs to pipe
#     settings = arch_data.get("gen_kwargs", {})
#     kwargs.update(settings)
#     # for classes in pipe_classes:
#     pipe = pipe_classes[0]
#     repo = repo_paths[0]
#     pkg = next(iter(import_pkg))
#     nfo(f"status mid-pipe: {repo}, {pipe}, {pkg}, {init_kwargs}, {kwargs}, ")
#     dbug(f"status mid-pipe: {repo_paths}, {pipe_classes}, {import_pkg}, {init_kwargs}, {kwargs}, ")
#     # current_kwargs = init_kwargs.get(pipe, init_kwargs) #
#     pipe = self._load_pipe(
#         pipe,
#         repo,
#         pkg,
#         **init_kwargs,  # apply kwargs per-pipe if possible
#     )
#     return (pipe, repo, pkg, kwargs)
