# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
import os
from typing import Optional, Callable, Any

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

    schnell = "schnell"
    dev = "dev"
    last_pipe: Callable = None

    def _load_pipe(self, pipe_stack: dict[str, Any]) -> Callable:
        stack_items = pipe_stack.get("main")
        pipe_obj = stack_items.pop("pipe_obj")
        model = stack_items.pop("model")
        pkg_name = stack_items.pop("pkg_name")

        model = model.model
        if pkg_name in ["diffusers", "transformers", "parler-tts"]:
            if os.path.isfile(model):
                pipe_method = pipe_obj.from_single_file
            else:
                pipe_method = pipe_obj.from_pretrained
                try:
                    pipe = pipe_method(model, use_safetensors=True, **stack_items)
                except (EnvironmentError, OSError):
                    pipe = pipe_method(model, **stack_items)
        if stack_items := pipe_stack.get("scheduler", {}):
            pipe_obj = stack_items.pop("pipe_obj", {})
            print(stack_items)
            pipe.scheduler = pipe_obj.from_config(pipe.scheduler.config, **stack_items)  # replace with setattr?
        if stack_items := pipe_stack.get("lora", {}):
            pipe_obj = stack_items.pop("pipe_obj")
            model = stack_items.pop("model")
            pipe.load_lora_weights = pipe_obj(model.model, weight_name=model.mir)
            if kwargs := stack_items.get("fuse", {}):
                pipe.fuse_lora(**kwargs)
        elif pkg_name == "mflux":
            base_name = self.schnell if self.schnell in model.base_name[0] else self.dev
            pipe_args = {"base_name": base_name, "local_path": model.path}
            if stack_items := pipe_stack.get("lora", {}):
                pipe(**pipe_args, **kwargs)
        elif pkg_name == "audiogen":
            return pipe.get_pretrained(model, **kwargs)
        elif pkg_name == "chroma":
            return pipe(**kwargs)
        return pipe
        if pipe is None:
            raise TypeError("Pipe should be Callable `class` object, not `None`")

    @debug_monitor
    @pipe_call
    async def create_pipeline(self, registry_entry: Callable, pkg_data: tuple[str], mir_db: MIRDatabase, **kwargs):
        """
        Build an inference pipe based on model type\n
        :param registry_entry: Data for the model
        :param pkg_data: Predetermined best package to run for this system
        :return: `tuple` constructed pipe, model/repo name `str`, arguments used in the pipe, and a `dict` of default settings
        """
        from importlib import import_module
        from nnll.metadata.helpers import make_callable

        # chip_stats = ChipStats()
        # stats = await chip_stats.show_stats(True)
        pkg_name = pkg_data[-1].value[1].lower()
        pkg_obj = import_module(pkg_name)
        if "." in pkg_data[1]:
            module_path = pkg_data[1].rsplit(".", 1)
            pipe_obj = make_callable(module_path[-1], f"{pkg_name}.{module_path[0]}")
        else:
            pipe_obj = getattr(pkg_obj, pkg_data[1])
        nfo(next(iter(pkg_data[0])))
        main_pipe = {"pipe_obj": pipe_obj, "model": registry_entry, "pkg_name": pkg_name} | kwargs
        if precision := registry_entry.modules.get(pkg_data[0], {}).get("precision"):
            precision = precision.rsplit(".", 1)
            dtype = mir_db.database[precision[0]][precision[1].upper()]["pkg"]["0"]  # get the precision class, currently assumed to be torch
            precision = next(iter(dtype["torch"]))  # add default?
            main_pipe.setdefault("torch_dtype", getattr(import_module("torch"), precision))
            if registry_entry.modules[pkg_data[0]].get("variant"):  # Added to skip non-available variants
                if variant := dtype["torch"].get(precision, {}):  # because of above, should only happen if the file is fp16 already!
                    main_pipe.setdefault(*variant.keys(), *variant.values())
        if isinstance(registry_entry.modules[pkg_data[0]].get(pkg_name), dict):
            if extra_kwargs := registry_entry.modules[pkg_data[0]][pkg_name].get(pipe_obj, {}):
                main_pipe = main_pipe | extra_kwargs
        pipe_stack = {"main": main_pipe}
        if registry_entry.modules[pkg_data[0]].get("scheduler"):  # a simple toggle for key presence
            if scheduler := mir_db.database[registry_entry.mir[0]][registry_entry.mir[1]].get("pipe_names", {}).get("scheduler", {}):
                if isinstance(scheduler[0], list):
                    scheduler = next(iter(scheduler))
                scheduler_data = mir_db.database[scheduler[0]][scheduler[1]]["pkg"]["0"]  # get the scheduler class, currently assumed to be in diffusers/transformers
                scheduler_path = scheduler_data.pop("module_path", {})
                scheduler_obj = make_callable(scheduler_data[pkg_name], f"{scheduler_path}")
                scheduler_pipe = {"pipe_obj": scheduler_obj}
                if kwargs := scheduler_data.pop("generation", {}):
                    scheduler_pipe.setdefault(*kwargs.keys(), *kwargs.values())
            pipe_stack.setdefault("scheduler", scheduler_pipe)
        # iterator/generator logic for processing stacks of loras will go here
        generation = registry_entry.modules[pkg_data[0]].get("generation", {})
        model_id = registry_entry.model
        # display here before possible crashes
        nfo(f"""status mid-pipe:
            id -{model_id}
            pipe_stack -{pipe_stack}
            generation - {generation} """)  # lora_pipe:{lora_pipe}

        dbug(f"status mid-pipe: {pipe_stack}")
        pipe = self._load_pipe(
            pipe_stack,
        )
        return (pipe, model_id, pkg_name, generation)

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
