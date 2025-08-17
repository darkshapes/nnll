# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
import os
from typing import Callable, List, Union

from nnll.configure.chip_stats import ChipStats
from nnll.metadata.json_io import read_json_file
from nnll.mir.maid import MIRDatabase
from nnll.monitor.console import nfo
from nnll.monitor.file import dbug, debug_monitor


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

    def _load_pipe(self, pipe_obj: str, model: str, pkg_name: str, **kwargs) -> Callable:
        model = model.model
        if pkg_name in ["diffusers", "transformers", "parler-tts"]:
            if os.path.isfile(model):
                try:
                    return pipe_obj.from_single_file(model, use_safetensors=True, **kwargs)
                except (EnvironmentError, OSError):
                    return pipe_obj.from_single_file(model, **kwargs)
            else:
                try:
                    return pipe_obj.from_pretrained(model, use_safetensors=True, **kwargs)
                except (EnvironmentError, OSError):
                    return pipe_obj.from_pretrained(model, **kwargs)
        elif pkg_name == "mflux":
            return pipe_obj(model_name="model", **kwargs)
        elif pkg_name == "audiogen":
            return pipe_obj.get_pretrained(model, **kwargs)
        elif pkg_name == "chroma":
            return pipe_obj(**kwargs)

        if pipe_obj is None:
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

        chip_stats = ChipStats()
        metrics = chip_stats.get_metrics()
        pkg_name = pkg_data[-1].value[1].lower()
        pkg_obj = import_module(pkg_name)
        if "." in pkg_data:
            split_pkg_data = pkg_data[1].rsplit(".", 1)
            pipe_obj = make_callable(split_pkg_data[-1], f"{pkg_name}.{split_pkg_data[0]}")
        else:
            pipe_obj = getattr(pkg_obj, pkg_data[1])

        model_id = registry_entry.model
        pipe_call = {"pipe_obj": pipe_obj, "model": registry_entry, "pkg_name": pkg_name} | kwargs
        if precision := registry_entry.modules[pkg_data[0]].get("precision"):
            precision = precision.rsplit(".", 1)
            dtype = mir_db.database[precision[0]][precision[1].upper()]["pkg"]["0"]  # get the precision class, currently assumed to be torch
            precision = next(iter(dtype["torch"]))
            pipe_call.setdefault("torch_dtype", getattr(import_module("torch"), precision))
            variant = dtype["torch"][precision]
            pipe_call.setdefault("torch_dtype", getattr(import_module("torch"), precision))
            variant = dtype["torch"][precision]  # this should only happen if the file is fp16 already!
            if variant:
                pipe_call.setdefault(*variant.keys(), *variant.values())
        if isinstance(registry_entry.modules[pkg_data[0]].get(pkg_name), dict):
            if extra_kwargs := registry_entry.modules[pkg_data[0]][pkg_name].get(pipe_obj):
                pipe_call = pipe_call | extra_kwargs | {"path": registry_entry.path}
        generation = registry_entry.modules[pkg_data[0]].get("generation", {})

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
