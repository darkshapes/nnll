#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0*/ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from typing import Any, Callable, List, get_type_hints


from nnll.mir.maid import MIRDatabase
from nnll.monitor.console import nfo
from nnll.monitor.file import dbuq


flatten_map: List[Any] = lambda nested, unpack: [element for iterative in getattr(nested, unpack)() for element in iterative]
flatten_map.__annotations__ = {"nested": List[str], "unpack": str}


class AutoPkg:
    def __init__(self) -> None:
        self.skip_series = [
            "info.lora",
            "info.vae",
            "ops.precision",
            "ops.scheduler",
            "info.encoder.tokenizer",
            "info.controlnet",
        ]
        self.skip_classes = [".gligen", "imagenet64"]
        self.skip_auto = ["AutoTokenizer", "AutoModel", "AutoencoderTiny", "AutoencoderKL", "AutoPipelineForImage2Image"]
        self.skip_types = ["int", "bool", "float", "Optional", "NoneType", "List", "UNet2DConditionModel"]
        self.mflux_tasks = ["Image", "Redux", "Kontext", "Depth", "Fill", "ConceptAttention", "ControlNet", "CavTon", "IC-Edit"]

    @staticmethod
    def show_diffusers_tasks(code_name: str, class_name: str | None = None) -> list[str]:
        """Return Diffusers task pipes based on package-specific query\n
        :param class_name: To find task pipes from a Diffusers class pipe, defaults to None
        :param code_name: To find task pipes from a Transformers class pipe, defaults to None
        :return: A list of alternate class pipelines derived from the specified class"""

        if class_name:
            from diffusers.pipelines.auto_pipeline import SUPPORTED_TASKS_MAPPINGS, _get_task_class

            alt_tasks = []
            for task_map in SUPPORTED_TASKS_MAPPINGS:
                task_class = _get_task_class(task_map, class_name, False)
                if task_class:
                    alt_tasks.append(task_class.__name__)
                    dbuq(task_class)
                for model_code, pipe_class_obj in task_map.items():
                    if code_name in model_code:
                        alt_tasks.append(pipe_class_obj.__name__)

        return alt_tasks

    @staticmethod
    def show_transformers_tasks(class_name: str | None = None, code_name: str | None = None) -> list[str]:
        """Retrieves a list of task classes associated with a specified transformer class.\n
        :param class_name: The name of the transformer class to inspect.
        :param pkg_type: The dependency for the module
        :param alt_method: Use an alternate method to return the classes
        :return: A list of task classes associated with the specified transformer."""

        if not code_name:
            from nnll.metadata.helpers import make_callable

            class_obj: Callable = make_callable(class_name, "transformers")
            class_module: Callable = make_callable(*class_obj.__module__.split(".", 1)[-1:], class_obj.__module__.split(".", 1)[0])
            task_classes = getattr(class_module, "__all__")
        elif code_name:
            from transformers.utils.fx import _generate_supported_model_class_names

            task_classes = _generate_supported_model_class_names(code_name)
        return task_classes

    async def detect_tasks(self, mir_db: MIRDatabase, field_name: str = "pkg") -> dict:
        """Detects and traces tasks MIR data\n
        :param mir_db:: An instance of MIRDatabase containing the database of information.
        :type mir_db: MIRDatabase
        :param field_name:  The name of the field in compatibility data to process for task detection, defaults to "pkg".
        :type field_name: str, optional
        :return: A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""

        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(tag for tag in self.skip_series if series.startswith(tag))
                and not any(tag for tag in self.skip_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        tasks_for_class = {"tasks": []}
                        for _, pkg_tree in field_data[field_name].items():
                            detected_tasks = await self.trace_tasks(pkg_tree=pkg_tree)
                            if detected_tasks:
                                for task in detected_tasks:
                                    if task not in tasks_for_class["tasks"]:
                                        tasks_for_class["tasks"].append(task)
                                data_tuple.append((*series.rsplit(".", 1), {compatibility: tasks_for_class}))

        return data_tuple

    async def detect_pipes(self, mir_db: MIRDatabase, field_name: str = "pkg") -> dict:
        """Detects and traces Pipes MIR data\n
        :param mir_db:: An instance of MIRDatabase containing the database of information.
        :type mir_db: MIRDatabase
        :param field_name:  The name of the field in compatibility data to process for task detection, defaults to "pkg".
        :type field_name: str, optional
        :return:A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""
        from nnll.metadata.helpers import make_callable

        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(series.startswith(tag) for tag in self.skip_series)
                and not any(tag for tag in self.skip_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        for _, pkg_tree in field_data[field_name].items():
                            if pkg_tree and next(iter(pkg_tree)) == "diffusers":
                                module_name = pkg_tree[next(iter(pkg_tree))]
                                dbuq(f"{module_name} pipe originator")
                                class_obj = make_callable(module_name, "diffusers")
                                pipe_args = get_type_hints(class_obj.__init__)
                                detected_pipe = await self.hyperlink_to_mir(pipe_args, series, mir_db)
                                data_tuple.append((*series.rsplit(".", 1), {compatibility: detected_pipe}))

        return data_tuple

    async def hyperlink_to_mir(self, pipe_args: dict, series: str, mir_db: MIRDatabase):
        """Maps pipeline components to MIR tags/IDs based on class names and roles.\n
        :param pipe_args: Dictionary of pipeline roles to their corresponding classes
        :param mir_db: MIRDatabase instance for querying tags/IDs
        :return: Dictionary mapping pipeline roles to associated MIR tags/IDs"""

        mir_tag: None | list[str] = None
        detected_links: dict[str, dict] = {"pipe_names": dict()}
        for pipe_role, pipe_class in pipe_args.items():
            if pipe_role in ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4", "prior_tokenizer"]:
                detected_links["pipe_names"].setdefault(pipe_role, ["info.encoder.tokenizer", series.rsplit(".", 1)[-1]])
                continue
            if not any(segment for segment in self.skip_types if pipe_class.__name__ == segment):
                mir_tag = None
                detected_links["pipe_names"][pipe_role] = []
                dbuq(f"pipe_class.__name__ {pipe_class.__name__} {pipe_class}")
                if pipe_class.__name__ in ["Union"]:
                    tags_or_classes = []
                    for union_class in pipe_class.__args__:
                        mir_tag = None
                        class_name = union_class.__name__
                        if not any(segment for segment in self.skip_types if class_name == segment):
                            mir_tag, class_name = await self.tag_class(pipe_class=union_class, pipe_role=pipe_role, series=series, mir_db=mir_db)
                            # mir_tag = mir_db.find_tag(field="tasks", target=class_name)
                            # dbuq(f"{mir_tag} {class_name}")
                            tags_or_classes.append(mir_tag if mir_tag else class_name)
                    detected_links["pipe_names"][pipe_role].extend(tags_or_classes)
                else:
                    mir_tag, class_name = await self.tag_class(pipe_class=pipe_class, pipe_role=pipe_role, series=series, mir_db=mir_db)
                    detected_links["pipe_names"][pipe_role] = mir_tag if mir_tag else [class_name]
                    mir_tag = None
                    class_name = None
        return detected_links

    async def tag_class(self, pipe_class: Callable, pipe_role: str, series: str, mir_db: MIRDatabase) -> tuple[str | None]:
        """Maps a class to MIR tags/IDs based on its name and role.\n
        :param pipe_class: Class to be mapped
        :param pipe_role: Role of the class in the pipeline
        :param series: Series identifier for the component
        :param mir_db: MIRDatabase instance for querying tags/IDs
        :return: Tuple containing MIR tag and class name"""

        from nnll.mir.tag import make_scheduler_tag

        mir_tag = None
        class_name = pipe_class.__name__
        if pipe_role in ["scheduler", "image_noising_scheduler", "prior_scheduler"]:
            sub_field = pipe_class.__module__.split(".")[0]
            scheduler_series, scheduler_comp = make_scheduler_tag(class_name)
            mir_tag = [f"ops.scheduler.{scheduler_series}", scheduler_comp]
            if not mir_db.database.get(mir_tag[0], {}).get(mir_tag[1]):
                mir_tag = mir_db.find_tag(field="pkg", target=class_name, sub_field=sub_field, domain="ops.scheduler")
            dbuq(f"scheduler {mir_tag} {class_name} {sub_field} ")
        elif pipe_role == "vae":
            sub_field = pipe_class.__module__.split(".")[0]
            mir_comp = series.rsplit(".", 1)[-1]
            dbuq(mir_comp)
            mir_tag = [mir_id for mir_id, comp_data in mir_db.database.items() if "info.vae" in mir_id and next(iter(comp_data)) == mir_comp]
            if mir_tag:
                mir_tag.append(mir_comp)  # keep mir tag as single list
            elif class_name != "AutoencoderKL":
                dbuq(pipe_class)
                mir_tag = mir_db.find_tag(field="pkg", target=class_name, sub_field=sub_field, domain="info.vae")
            dbuq(f"vae {mir_tag} {class_name} {sub_field} ")
        else:
            mir_tag = mir_db.find_tag(field="tasks", target=class_name)
        return mir_tag, class_name

    async def trace_tasks(self, pkg_tree: dict[str, str | int | list[str | int]]) -> List[str]:
        """Trace tasks for a given MIR entry.\n
        :param entry: The object containing the model information.
        :return: A sorted list of tasks applicable to the model."""

        from nnll.tensor_pipe.deconstructors import get_code_names

        preformatted_task_data = None
        filtered_tasks = None
        snip_words: set[str] = {"load_tf_weights_in"}
        package_name = next(iter(pkg_tree))
        dbuq(pkg_tree)
        class_name = pkg_tree[package_name]
        dbuq(f"{package_name}, {class_name}")
        if class_name not in self.skip_auto:
            if isinstance(class_name, dict):
                class_name = next(iter(list(class_name)))
            if package_name == "transformers":
                preformatted_task_data = self.show_transformers_tasks(class_name=class_name)
            elif package_name == "diffusers":
                code_name = get_code_names(class_name, package_name)
                preformatted_task_data = self.show_diffusers_tasks(code_name=code_name, class_name=class_name)
                preformatted_task_data.sort()
            elif package_name == "mflux":
                preformatted_task_data = self.mflux_tasks
            if preformatted_task_data:
                filtered_tasks = [task for task in preformatted_task_data for snip in snip_words if snip not in task]
                return filtered_tasks  # package_name, class_name


def main(mir_db: MIRDatabase = None):
    """Parse arguments to feed to dict header reader"""
    import argparse
    import asyncio
    from nnll.mir.automata import assimilate
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scrape the task classes from currently installed libraries and attach them to an existing MIR database.\nOffline function.",
            usage="mir-tasks",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = AutoPkg()
    task_tuple = asyncio.run(auto_pkg.detect_tasks(mir_db))

    assimilate(mir_db, [task for task in task_tuple])

    mir_db.write_to_disk()
    return mir_db


def pipe(mir_db: MIRDatabase = None):
    import argparse
    import asyncio
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Infer pipe components from Diffusers library and attach them to an existing MIR database.\nOffline function.",
            usage="mir-pipe",
            epilog="Can be run automatically with `python -m nnll.mir.maid` Should only be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    from nnll.mir.automata import assimilate

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = AutoPkg()
    pipe_tuple = asyncio.run(auto_pkg.detect_pipes(mir_db))
    assimilate(mir_db, [pipe for pipe in pipe_tuple])
    mir_db.write_to_disk()
    return mir_db


if __name__ == "__main__":
    pipe()
