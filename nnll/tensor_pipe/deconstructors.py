# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

# """动态函数工厂"""
from typing import Callable, Dict, List
from importlib import import_module
# from mir.mir_maid import MIRDatabase

from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS
# from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

short_name = next(iter(x for x in DIFFUSERS_DEFAULT_PIPELINE_PATHS if "refiner" in x))

# infer_diffusers_model_type


def _get_task_pipe(pkg_name: dict, i2i: bool = False) -> list[Callable]:
    """Convert normal diffusers pipe to a task-specific pipe\n
    :return: a list of Callable element to import
    """
    from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

    task = "IMAGE2IMAGE" if i2i else "INPAINT"
    task_pipe = _get_task_class(next(iter(x for x in SUPPORTED_TASKS_MAPPINGS if task in x)), pkg_name.get("diffusers"))
    return [task_pipe]


def root_class(init_module: Callable) -> Dict[str : List[str, str]]:
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


def find_pipe(query: str):
    from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING


# mir_db = MIRDatabase()
# package_tree = {}
# pipe_list = ["FluxPipeline", "StableDiffusion3Pipeline", "StableDiffusionPipeline", "StableDiffusionXLPipeline", "StableCascadeCombinedPipeline"]
# for i in pipe_list:
#     pkg = import_module("diffusers")
#     init_module = getattr(pkg, i)
#     package_tree = root_class(init_module=init_module)
#     for component, modules in package_tree.items()
#         series_name = modules[-1]
#         for minor in ["Discrete", "Scheduler", "Multistep", "Solver",]:
#                 series_name = series_name.replace(minor, "")
#         mir_db.database[f"ops.{component}")
#                 domain="ops",
#                 arch=component,
#                 series=series_name.lower(),
#                 comp="diffusers",
#                 package={modules[0]:modules[1:]},
#             )
#         )


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

# check = {LibType.HUB}


# def filter_dependencies(model_path: str, mir_db: Callable = MIRDatabase()):
#     mir_tag = mir_db.find_path(field="repo", target=model_path)


#     if "".join(mir_tag).find("flux") > 0 and LibType.check_type("MFLUX") and "MPS" in ChipType._show_ready("mps"):
#         nfo("HAS MFLUX")
#     # pylint:disable=no-member
#     return mir_tag
#     # if LibType.HUB:


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
