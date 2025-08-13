# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
import os
from typing import Callable, List, Union

from nnll.monitor.console import nfo
from nnll.monitor.file import dbug, debug_monitor
from nnll.mir.maid import MIRDatabase


@debug_monitor
def pipe_call(func):
    """Decorator for Diffusers pipes to combine arguments"""
    import inspect
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_params = inspect.signature(func).parameters
        args_dict = dict(zip(func_params, args))
        kwargs.update({k: v for k, v in args_dict.items() if v is not None and k not in kwargs})
        return func(**kwargs)

    return wrapper


class ConstructPipeline:
    """Build and configure Diffusers pipelines"""

    last_pipe: Callable = None

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

    # def _get_sub_cls_locs(self, init_module):
    #     import inspect

    #     signature = inspect.signature(init_module.__init__)
    #     class_sigs = {}
    #     for folder, param in signature.parameters.items():
    #         if folder != "self":
    #             sub_module = str(param.annotation).split("'")
    #             if len(sub_module) > 1 and sub_module[1] not in ["bool", "int", "float", "complex", "str", "list", "tuple", "dict", "set"]:
    #                 class_sigs.setdefault(folder, sub_module[1].split("."))
    #     return class_sigs

    # def _get_task_pipe(self, pkg_name: dict, inpaint: bool = False, i2i: bool = False) -> list["Callable"]:
    #     """Convert normal pipe to task-specific pipe\n
    #     :return: a list of Callable element to import
    #     """
    #     from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

    #     task = "IMAGE2IMAGE" if i2i else "INPAINT"
    #     task_pipe = _get_task_class(next(iter(x for x in SUPPORTED_TASKS_MAPPINGS if task in x)), pkg_name.get("diffusers"))
    #     return [task_pipe]

    def _load_pipe(self, pipe_obj: str, model: str, pkg_name: str, **kwargs) -> Callable:
        if pkg_name in ["diffusers", "transformers", "parler-tts"]:
            if os.path.isfile(model):
                pipe = pipe_obj.from_single_file(model, **kwargs)
                return pipe
            else:
                pipe = pipe_obj.from_pretrained(model, **kwargs)
                return pipe
        elif pkg_name == "audiogen":
            pipe_class = pipe_obj.get_pretrained(model, **kwargs)
            return pipe
        if pipe_class is None:
            raise TypeError("Pipe should be Callable `class` object, not `None`")

    @debug_monitor
    @pipe_call
    def create_pipeline(self, registry_entry: Callable, pkg_data: tuple[str], mir_db: MIRDatabase, **kwargs):
        """
        Build an inference pipe based on model type\n
        :param registry_entry: Data for the model
        :param pkg_data: Predetermined best package to run for this system
        :return: `tuple` constructed pipe, model/repo name `str`, arguments used in the pipe, and a `dict` of default settings
        """
        from importlib import import_module

        from nnll.metadata.helpers import make_callable

        pkg_name = pkg_data[1].value[1].lower()
        pkg_obj = import_module(pkg_name)
        if "." in pkg_data:
            split_pkg_data = pkg_data[0].rsplit(".", 1)
            pipe_obj = make_callable(split_pkg_data[-1], f"{pkg_name}.{split_pkg_data[0]}")
        else:
            pipe_obj = getattr(pkg_obj, pkg_data[0])

        model_id = registry_entry.model
        package_keys = registry_entry.modules.keys()
        pipe_call = {"pipe_obj": pipe_obj, "model": model_id, "pkg_name": pkg_name} | kwargs
        pkg_index = [i for i in package_keys if pkg_name in registry_entry.modules[i]]

        if precision := registry_entry.modules[pkg_index[0]].get("precision"):
            precision = precision.rsplit(".", 1)
            dtype = mir_db.database[precision[0]][precision[1].upper()]["pkg"]["0"]
            precision = next(iter(dtype["torch"]))
            pipe_call.setdefault("torch_dtype", getattr(import_module("torch"), precision))
            variant = dtype["torch"][precision]
            if variant:
                pipe_call.setdefault(*variant.keys(), *variant.values())
        generation = registry_entry.modules[pkg_index[0]].get("generation",{})

        nfo(f"status mid-pipe: {model_id}, {pipe_obj}, {pipe_call}, {generation} ")
        dbug(f"status mid-pipe: {model_id}, {pipe_obj}, {pipe_call}, {generation}, ")
        pipe = self._load_pipe(
            **pipe_call,
        )
        return (pipe, model_id, pipe_call, generation)


    # def add_lora(self, pipe: Callable, lora_repo: str, init_kwargs: dict, scheduler_data=None, scheduler_kwargs=None):
    #     if scheduler_data:
    #         import_pkg = scheduler_data["pkg"]
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
