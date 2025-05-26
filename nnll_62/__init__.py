### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

"""动态函数工厂"""

# pylint: disable=unsubscriptable-object, import-outside-toplevel, unused-argument, line-too-long
import os
from typing import Callable
from nnll_01 import debug_monitor, dbug, nfo
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

    def _get_module(self, import_pkg: dict[str, list[str]]) -> list[Callable]:
        """Accept two lists of importable dependencies and modules\n
        :param pkg_name: Main external dependencies
        :param module_path: Sub-modules of the main dependency
        :return: A list of callable statements
        """
        import importlib

        import_modules = []
        for name, module in import_pkg.items():
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

    def _load_pipe(self, pipe_class: str, repo: str, import_pkg: str, **kwargs) -> Callable:
        if import_pkg in ["diffusers", "transformers", "parler-tts"]:
            if os.path.isfile(repo):
                pipe = pipe_class.from_single_file(repo, **kwargs)
                return pipe
            # from config
            else:
                pipe = pipe_class.from_pretrained(repo, **kwargs)
                return pipe
        elif import_pkg == "audiogen":
            pipe_class = pipe_class.get_pretrained(repo, **kwargs)
            return pipe
        if pipe_class is None:
            raise TypeError("Pipe should be Callable `class` object, not `None`")

    def add_lora(self, pipe, lora):
        series = lora[0]
        arch_data = self.construct[series].get(lora[1])
        nfo(arch_data)
        lora_repo = next(iter(arch_data["repo"]))  # <- user location here OR this
        scheduler = self.construct[series]["[init]"].get("scheduler")
        if scheduler:
            sched = self.construct[scheduler]["[init]"]
            import_pkg = sched["dep_pkg"]
            scheduler_class = self._get_module(import_pkg)
            scheduler_kwargs = self.construct[series]["[init]"].get("scheduler_kwargs")
            pipe.scheduler = scheduler_class[0]({**scheduler_kwargs})
            nfo(f"mid sched {sched}, {scheduler_class}, {series}")
        nfo(f"status mid-lora: {lora}, {arch_data}, {pipe}, {scheduler}, ")
        init_kwargs = arch_data.get("init_kwargs")
        fuse = 0
        if init_kwargs is not None:
            fuse = init_kwargs.get("fuse", 0)
        pipe.load_lora_weights(lora_repo, adapter_name=os.path.basename(lora_repo))
        if fuse:
            pipe.fuse_lora(adapter_name=os.path.basename(lora_repo), lora_scale=fuse)
            pipe.unload_lora_weights()
        return pipe

    @debug_monitor
    @pipe_call
    def create_pipeline(self, architecture: list[str], *args, granular: bool = False, lora: list | None = None, **kwargs):
        """
        Build an inference pipe based on model type\n
        :param architecture: Identifier of model architecture
        :return: `tuple` constructed pipe, model/repo name `str`, and a `dict` of default settings
        """
        series = architecture[0]
        arch_data = self.construct[series].get(architecture[1])
        # if series.split(".")[1] == "lora":
        #     scheduler = arch_data.get("solver", 0)
        init_modules = self.construct[series]["[init]"]
        # granular should be set by spec check
        import_pkg = init_modules["dep_pkg"]  # a dictionary of dependencies

        # handle alternate external dependencies that are not installed
        # if arch_data.get("dep_alt", 0):
        #     import_pkg.update(arch_data["dep_alt"])

        pipe_classes = self._get_module(import_pkg)  # now a list of classes
        if granular and ("diffusers" in import_pkg or "transformers" in import_pkg):
            # break down pipeline for low-spec machine
            sub_cls_locs = self._get_sub_cls_locs(pipe_classes[-1])
            # for folder, package in sub_cls_locs:
            # self._get_module(package[:1], package[-1:])
            # self.construct.find_path()

        repo_paths = arch_data.get("repo")
        init_kwargs = arch_data.get("init_kwargs", {})
        init_kwargs.update(kwargs)  # add user kwargs to pipe
        settings = arch_data.get("gen_kwargs", {})
        kwargs.update(settings)
        # for classes in pipe_classes:
        pipe = pipe_classes[0]
        repo = repo_paths[0]
        pkg = next(iter(import_pkg))
        nfo(f"status mid-pipe: {repo}, {pipe}, {pkg}, {init_kwargs}, {kwargs}, ")
        dbug(f"status mid-pipe: {repo_paths}, {pipe_classes}, {import_pkg}, {init_kwargs}, {kwargs}, ")
        # current_kwargs = init_kwargs.get(pipe, init_kwargs) #
        pipe = self._load_pipe(
            pipe,
            repo,
            pkg,
            **init_kwargs,  # apply kwargs per-pipe if possible
        )
        if lora:
            pipe = self.add_lora(pipe, lora)
        return (pipe, repo, pkg, kwargs)
