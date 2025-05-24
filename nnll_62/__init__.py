### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long

from typing import Callable
from nnll_01 import debug_monitor, dbug
from nnll_60.mir_maid import MIRDatabase


@debug_monitor
def pipe_call(func):
    """Decorator for Diffusers pipes to combine arguments"""
    from functools import wraps
    import inspect

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_params = inspect.signature(func).parameters
        args_dict = dict(zip(func_params, args))
        kwargs.update({k: v for k, v in args_dict.items() if v is not None and k not in kwargs})
        return func(**kwargs)

    return wrapper


class ConstructPipeline:
    """Build and configure Diffusers pipelines"""

    @debug_monitor
    @pipe_call
    def create_pipeline(self, architecture: list[str], *args, lora: list | None = None, **kwargs):
        """
        Build an inference pipe based on model type\n
        :param architecture: Identifier of model architecture
        :return: `tuple` constructed pipe, model/repo name `str`, and a `dict` of default settings
        """
        import os
        import importlib

        def get_module(pkg_name: str, module_path: str) -> Callable:
            pkg = importlib.import_module(pkg_name)
            return getattr(pkg, ".".join(module_path)) if module_path is not None and len(module_path) > 1 else getattr(pkg, next(iter(module_path)))

        def load_pipe(pipe_class: str, repo_path: str, **kwargs) -> Callable:
            return pipe_class.from_single_file(repo_path, **kwargs) if os.path.isfile(repo_path) else pipe_class.from_pretrained(repo_path, **kwargs)

        data = MIRDatabase()
        construct = data.read_from_disk()
        dbug(construct)
        series = architecture[0]
        arch_data = construct[series].get(architecture[1])
        init_modules = construct[series]["[init]"]
        repo = arch_data.get("repo")

        import_pkg = next(iter(init_modules["dep_pkg"]))
        import_module = arch_data.get("module_alt") or init_modules.get("module_path")
        pipe_class = get_module(import_pkg, import_module)

        init_kwargs = arch_data.get("init_kwargs", {})
        init_kwargs.update(kwargs)
        settings = arch_data.get("gen_kwargs", {})
        kwargs.update(settings)

        pipe = load_pipe(pipe_class, repo, **init_kwargs)

        if series.split(".")[1] == "lora":
            scheduler = arch_data.get("solver", 0)
            if scheduler:
                sched = construct[scheduler].get(series)
                scheduler_class = get_module(import_pkg, sched)
                pipe.scheduler = scheduler_class(arch_data.get("solver_kwargs"), {})

            fuse = arch_data.get("fuse")
            pipe.load_lora_weights(repo, adapter_name=os.path.basename(repo))
            if fuse:
                print(os.path.basename(repo), fuse)
                pipe.fuse_lora(adapter_name=os.path.basename(repo), lora_scale=fuse)
                pipe.unload_lora_weights()

        return (pipe, repo, kwargs)

    # def create_pipeline(self, architecture: list[str], *args, lora: list | None = None, **kwargs):
    #     """
    #     Build an inference pipe based on model type\n
    #     :param architecture: Identifier of model architecture
    #     :param call: Return the pipe components or the called function, defaults to False
    #     :return: `tuple` constructed pipe, model/repo name, and default settings
    #     """
    #     import os
    #     import importlib

    #     data = MIRDatabase()
    #     construct = data.read_from_disk()
    #     dbug(construct)
    #     series = architecture[0]
    #     arch_data = construct[series].get(architecture[1])
    #     init_modules = construct[series]["[init]"]
    #     repo = arch_data.get("repo")
    #     import_pkg = next(iter(init_modules["dep_pkg"]))
    #     pkg = importlib.import_module(import_pkg)
    #     import_module = arch_data.get("module_alt", None)
    #     pipe_class = getattr(pkg, ".".join(import_module)) if len(import_module) > 1 else getattr(pkg, import_module)
    #     init_kwargs = arch_data.get("init_kwargs", {})
    #     init_kwargs.update(kwargs)
    #     # do a barrel roll (and spec check)
    #     # _attn_implementation='flash_attention_2',
    #     settings = arch_data.get("gen_kwargs", {})
    #     kwargs.update(settings)
    #     if os.path.isfile(repo):
    #         pipe = pipe_class.from_single_file(repo, **init_kwargs)
    #     else:
    #         pipe = pipe_class.from_pretrained(repo, **init_kwargs)
    #     if series.split(".")[1] == "lora":
    #         scheduler = arch_data.get("solver", 0)
    #         if scheduler:
    #             sched = construct[scheduler].get(series)
    #             scheduler_class = getattr(pkg, sched)
    #             getattr(pkg, ".".join(sched)) if len(sched) > 1 else getattr(pkg, sched)
    #             pipe.scheduler = scheduler_class(arch_data.get("solver_kwargs"))
    #         fuse = arch_data.get("fuse")
    #         pipe.load_lora_weights(repo, adapter_name=os.path.basename(repo))
    #         if fuse:
    #             print(os.path.basename(repo), fuse)
    #             pipe.fuse_lora(adapter_name=os.path.basename(repo), lora_scale=fuse)
    #             pipe.unload_lora_weights()
    #     # repo = arch_data.get("local") # here we can pull user-defined settings
    #     return (pipe, repo, kwargs)
