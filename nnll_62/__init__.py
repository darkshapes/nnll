### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
import os
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

    data = MIRDatabase()
    construct = data.read_from_disk()

    def _get_module(self, pkg_name: dict[str, list[str]]) -> list[Callable]:
        """Accept two lists of importable dependencies and modules\n
        :param pkg_name: Main external dependencies
        :param module_path: Sub-modules of the main dependency
        :return: A list of callable statements
        """
        import importlib

        import_modules = []
        for name, module in pkg_name.items():
            try:
                pkg = importlib.import_module(name)
            except (ImportError, AttributeError):
                continue
            else:
                if not next(iter(module)):
                    return
                elif len(module) > 1:
                    import_modules.append(getattr(pkg, ".".join(module)))
                else:
                    import_modules.append(getattr(pkg, next(iter(module))))
        return import_modules

    def _get_sub_cls_locs(self, init_module):
        import inspect

        signature = inspect.signature(init_module.__init__)
        class_sigs = {}
        for folder, param in signature.parameters.items():
            if folder != "self":
                sub_module = str(param.annotation).split("'")
                if len(sub_module) > 1 and sub_module[1] not in ["bool", "int", "float", "complex", "str", "list", "tuple", "dict", "set"]:
                    class_sigs.setdefault(folder, sub_module[1].split("."))
        return class_sigs

    def _get_task_pipe(self, pkg_name: dict, inpaint: bool = False, i2i: bool = False) -> list["Callable"]:
        """Convert normal pipe to task-specific pipe\n
        :return: a list of Callable element to import
        """
        from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

        task = "IMAGE2IMAGE" if i2i else "INPAINT"
        task_pipe = _get_task_class(next(iter(x for x in SUPPORTED_TASKS_MAPPINGS if task in x)), pkg_name.get("diffusers"))
        return [task_pipe]

    def _load_pipe(self, pipe_class: str, repo_path: str, import_pkg: str, **kwargs) -> Callable:
        if import_pkg in ["diffusers", "transformers", "parler-tts"]:
            if os.path.isfile(repo_path):
                return pipe_class.from_single_file(repo_path, **kwargs)
                # from config
            return pipe_class.from_pretrained(repo_path, **kwargs)
        elif import_pkg == "audiogen":
            return pipe_class.get_pretrained(repo_path, **kwargs)

    @debug_monitor
    @pipe_call
    def create_pipeline(self, architecture: list[str], *args, granular: bool = False, lora: list | None = None, **kwargs):
        """
        Build an inference pipe based on model type\n
        :param architecture: Identifier of model architecture
        :return: `tuple` constructed pipe, model/repo name `str`, and a `dict` of default settings
        """

        dbug(self.construct)
        series = architecture[0]
        arch_data = self.construct[series].get(architecture[1])
        init_modules = self.construct[series]["[init]"]
        # granular should be set by spec check
        import_pkg = init_modules["dep_pkg"]
        if arch_data.get("dep_alt", 0):
            import_pkg.update(arch_data["dep_alt"])
        init_pkg = self._get_module(init_modules["dep_pkg"])
        if granular and ("diffusers" in import_pkg or "transformers" in import_pkg):
            sub_cls_locs = self._get_sub_cls_locs(init_pkg[-1])
            for folder, package in sub_cls_locs:
                self._get_module(package[:1], package[-1:])
            # self.construct.find_path()
        repo = arch_data.get("repo")
        pipe_class = self._get_module(import_pkg)
        init_kwargs = arch_data.get("init_kwargs", {})
        init_kwargs.update(kwargs)
        settings = arch_data.get("gen_kwargs", {})
        kwargs.update(settings)

        pipe = self._load_pipe(pipe_class, repo, import_pkg, **init_kwargs)

        if series.split(".")[1] == "lora":
            scheduler = arch_data.get("solver", 0)
            if scheduler:
                sched = construct[scheduler].get(series)
                scheduler_class = self._get_module(import_pkg, sched)
                pipe.scheduler = scheduler_class(arch_data.get("solver_kwargs"), {})

            fuse = arch_data.get("fuse")
            pipe.load_lora_weights(repo, adapter_name=os.path.basename(repo))
            if fuse:
                print(os.path.basename(repo), fuse)
                pipe.fuse_lora(adapter_name=os.path.basename(repo), lora_scale=fuse)
                pipe.unload_lora_weights()

        return (pipe, repo, import_pkg, kwargs)
