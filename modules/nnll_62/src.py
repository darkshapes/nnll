### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint:disable=import-outside-toplevel
# pylint:disable=unused-argument
# pylint:disable=line-too-long

import os
import inspect
from functools import wraps
from typing import LiteralString

import torch
import diffusers


from modules.nnll_60.src import JSONCache, CONFIG_PATH_NAMED

device: str = "mps"
precision: torch.dtype = torch.float16
variant: LiteralString = """fp16"""
config_file = JSONCache(CONFIG_PATH_NAMED)


def pipe_call(func):
    """Decorator for Diffusers pipes to combine arguments"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_params = inspect.signature(func).parameters
        args_dict = dict(zip(func_params, args))
        kwargs.update({k: v for k, v in args_dict.items() if v is not None and k not in kwargs})
        return func(**kwargs)

    return wrapper


class ConstructPipeline:
    """Build and configure Diffusers pipelines"""

    @config_file.decorator
    def __init__(self, *args, **kwargs):
        """Encapsulate the config file for later use"""
        self.construct = kwargs.get("data", None)

    @pipe_call
    def create_pipeline(self, architecture, *args, **kwargs):
        """
        Build a diffusers pipe based on model type\n
        :return: `tuple` constructed pipe, model/repo name, and default settings
        """

        arch_data = self.construct[architecture]
        # repo = arch_data.get("local")
        repo = None
        if not repo:
            repo = arch_data.get("repo")
        pipe_class = getattr(diffusers, arch_data["pipe_name"])
        pipe_kwargs = arch_data.get("pipe_kwargs", {})
        pipe_kwargs.update(kwargs)

        if os.path.isfile(repo):
            pipe = pipe_class.from_single_file(repo, **pipe_kwargs)
        else:
            pipe = pipe_class.from_pretrained(repo, **pipe_kwargs)
            raise NotImplementedError("Support for only from_pretrained and from_single_file")

        settings = arch_data.get("defaults", {})
        kwargs.update(settings)

        return pipe, repo, kwargs

    @pipe_call
    def add_lora(self, lora, architecture, pipe, *args, **kwargs):
        """
        Add a LoRA to the diffusers pipe\n
        :return: `tuple` constructed pipe, model/repo name, and default settings
        """
        lora_data = self.construct[lora]
        arch_data = lora_data.get(architecture)
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
        return pipe, repo, kwargs
