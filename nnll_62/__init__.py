### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint:disable=import-outside-toplevel
# pylint:disable=unused-argument
# pylint:disable=line-too-long


from nnll_01 import debug_monitor, dbug
from nnll_60 import JSONCache, CONFIG_PATH_NAMED

config_file = JSONCache(CONFIG_PATH_NAMED)


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
    def create_pipeline(self, architecture: str, *args, join: bool = True, **kwargs):
        """
        Build a diffusers pipe based on model type\n
        :param architecture: Identifier of model architecture
        :param call: Return the pipe components or the called function, defaults to False
        :return: `tuple` constructed pipe, model/repo name, and default settings
        """
        import os
        import diffusers

        @config_file.decorator
        def _read_data(data: dict = None):
            return data

        construct = _read_data()

        dbug(construct)
        arch_data = construct[architecture]  # pylint:disable = unsubscriptable-object
        # repo = arch_data.get("local") # here we can pull user-defined settings
        repo = None
        if not repo:
            repo = arch_data.get("repo")
        pipe_class = getattr(diffusers, arch_data["pipe_name"])
        pipe_kwargs = arch_data.get("pipe_kwargs", {})
        pipe_kwargs.update(kwargs)
        settings = arch_data.get("defaults", {})
        kwargs.update(settings)

        if join:
            if os.path.isfile(repo):
                pipe = pipe_class.from_single_file(repo, **pipe_kwargs)
            else:
                pipe = pipe_class.from_pretrained(repo, **pipe_kwargs)
            return (pipe, repo, kwargs)
        if os.path.isfile(repo):
            return (pipe_class, "from_single_file", pipe_kwargs, repo, kwargs)
        return (pipe_class, "from_pretrained", pipe_kwargs, repo, kwargs)

        # raise NotImplementedError("Support for only from_pretrained and from_single_file")

    @debug_monitor
    @pipe_call
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
