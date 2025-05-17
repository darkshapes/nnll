### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint:disable=import-outside-toplevel
# pylint:disable=unused-argument
# pylint:disable=line-too-long


from nnll_01 import debug_monitor, dbug
from nnll_60 import JSONCache, CONFIG_PATH_NAMED

config_file = JSONCache(CONFIG_PATH_NAMED)


# @debug_monitor
# async def pipe_call(func):
#     """Decorator for Diffusers pipes to combine arguments"""
#     from functools import wraps
#     import inspect

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         func_params = inspect.signature(func).parameters
#         args_dict = dict(zip(func_params, args))
#         kwargs.update({k: v for k, v in args_dict.items() if v is not None and k not in kwargs})
#         return func(**kwargs)

#     return wrapper


class ConstructPipeline:
    """Build and configure Diffusers pipelines"""

    # @debug_monitor
    # @pipe_call
    def create_pipeline(self, architecture: str, *args, **kwargs):
        """
        Build a diffusers pipe based on model type\n
        :return: `tuple` constructed pipe, model/repo name, and default settings
        """
        import os
        import diffusers
        from importlib import import_module

        @config_file.decorator
        def _read_data(data: dict = None):
            return data

        construct = _read_data()

        dbug(construct)
        arch_data = construct[architecture]  # pylint:disable = unsubscriptable-object
        # repo = arch_data.get("local")
        repo = None
        if not repo:
            repo = arch_data.get("repo")
        pipe_class = getattr(diffusers, arch_data["pipe_name"])
        dynamic_module = import_module(f"{pipe_class}", diffusers)
        pipe_kwargs = arch_data.get("pipe_kwargs", {})
        pipe_kwargs.update(kwargs)

        if os.path.isfile(repo):
            pipe = dynamic_module.from_single_file(repo, **pipe_kwargs)
        else:
            dbug("pipe_class_test : {pipe_class}")
            # from diffusers import CogView3PlusPipeline

            pipe = dynamic_module.from_pretrained(repo, **pipe_kwargs)
            # pipe = pipe_class.from_pretrained(repo, **pipe_kwargs)
            # raise NotImplementedError("Support for only from_pretrained and from_single_file")

        settings = arch_data.get("defaults", {})
        kwargs.update(settings)

        return (pipe, repo, kwargs)

    # @debug_monitor
    # @pipe_call
    def add_lora(self, lora, architecture, pipe, *args, **kwargs):
        """
        Add a LoRA to the diffusers pipe\n
        :return: `tuple` constructed pipe, model/repo name, and default settings
        """
        import os
        import diffusers
        from pathlib import Path

        @config_file.decorator
        def _read_data(data: dict = None):
            return data

        construct = _read_data()

        lora_data = construct[lora]  # pylint:disable = unsubscriptable-object

        arch_data = lora_data.get(Path(architecture).suffix[1:])
        solver = lora_data.get("solver")
        if solver:
            scheduler_class = getattr(diffusers, solver)
            pipe.scheduler = scheduler_class(lora_data.get("solver_kwargs"))
        # repo = arch_data.get("local")
        repo = None
        if not repo:
            repo = arch_data.get("repo")
        fuse = arch_data.get("fuse")
        pipe.load_lora_weights(repo, adapter_name=os.path.basename(repo))
        if fuse:
            print(os.path.basename(repo), fuse)
            pipe.fuse_lora(adapter_name=os.path.basename(repo), lora_scale=fuse)
            pipe.unload_lora_weights()
        return (pipe, repo, kwargs)
