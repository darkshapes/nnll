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
        pass

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
        :return: A list of task classes associated with the specified transformer.
        """
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
        :return:A dictionary mapping series names to their respective compatibility and traced tasks.
        :rtype: dict"""

        avoid_classes = [".gligen", "imagenet64"]
        avoid_series = ["info.lora", "info.vae", "ops.precision", "ops.scheduler"]
        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(series.startswith(tag) for tag in avoid_series)
                and not any(tag for tag in avoid_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        tasks_for_class = {"tasks": []}
                        for index, pkg_tree in field_data[field_name].items():
                            task_data = await self.trace_tasks(pkg_tree=pkg_tree)
                            if task_data:
                                package_name, class_name, detected_tasks = task_data
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

        avoid_classes = [".gligen", "imagenet64"]
        avoid_series = ["info.lora", "info.vae", "ops.precision", "ops.scheduler"]
        data_tuple = []
        for series, compatibility_data in mir_db.database.items():
            if (
                series.startswith("info.")  # formatting comment
                and not any(series.startswith(tag) for tag in avoid_series)
                and not any(tag for tag in avoid_classes if tag in series)
            ):
                for compatibility, field_data in compatibility_data.items():
                    if field_data and field_data.get(field_name, {}).get("0"):
                        for _, pkg_tree in field_data[field_name].items():
                            if pkg_tree and next(iter(pkg_tree)) == "diffusers":
                                module_name = pkg_tree[next(iter(pkg_tree))]
                                dbuq(f"{module_name} pipe originator")
                                class_obj = make_callable(module_name, "diffusers")
                                pipe_args = get_type_hints(class_obj.__init__)
                                detected_pipe = await self.hyperlink_to_mir(pipe_args, mir_db)
                                data_tuple.append((*series.rsplit(".", 1), {compatibility: detected_pipe}))

        return data_tuple

    async def hyperlink_to_mir(self, pipe_args: dict, mir_db: MIRDatabase):
        """_summary_\n
        :param pipe_args: _description_
        :param mir_db: _description_
        :return: _description_"""
        excluded_types = ["int", "bool", "float", "Optional", "NoneType", "List", "UNet2DConditionModel"]
        mir_tag: None | list[str] = None
        detected_links: dict[str, dict] = {"pipe_names": dict()}
        for pipe_role, pipe_class in pipe_args.items():
            if not any(segment for segment in excluded_types if pipe_class.__name__ == segment):
                detected_links["pipe_names"][pipe_role] = []
                dbuq(f"pipe_class.__name__ {pipe_class.__name__} {pipe_class}")
                if pipe_class.__name__ in ["Union"]:
                    for union_class in pipe_class.__args__:
                        mir_tag = None
                        class_name = union_class.__name__
                        if not any(segment for segment in excluded_types if class_name == segment):
                            mir_tag = mir_db.find_tag(field="tasks", target=class_name)
                            dbuq(f"{mir_tag} {class_name}")
                            detected_links["pipe_names"][pipe_role].append(mir_tag if mir_tag else class_name)
                else:
                    class_name = pipe_class.__name__
                    if pipe_role == "scheduler":
                        sub_field = pipe_class.__module__.split(".")[0]
                        # print(class_name)
                        mir_tag = mir_db.find_tag(field="pkg", target=class_name, sub_field=sub_field, domain="ops.scheduler")
                        dbuq(f"scheduler {mir_tag} {class_name} {sub_field} ")
                    else:
                        mir_tag = mir_db.find_tag(field="tasks", target=class_name)
                    # dbuq(f"{mir_tag} {class_name}")
                    detected_links["pipe_names"][pipe_role].append(mir_tag if mir_tag else class_name)
        return detected_links

    async def trace_tasks(self, pkg_tree: dict[str, str | int | list[str | int]]) -> List[str]:
        """Trace tasks for a given MIR entry.\n
        :param entry: The object containing the model information.
        :return: A sorted list of tasks applicable to the model."""

        from nnll.tensor_pipe.deconstructors import get_code_names

        preformatted_task_data = None
        filtered_tasks = None
        snip_words: set[str] = {"load_tf_weights_in"}  # "PreTrained", "ForConditionalGeneration",
        package_name = next(iter(pkg_tree))
        class_name = pkg_tree[package_name]
        dbuq(f"{package_name}, {class_name}")
        if class_name not in ["AutoTokenizer", "AutoModel", "AutoencoderTiny", "AutoencoderKL", "AutoPipelineForImage2Image"]:
            if isinstance(class_name, dict):
                class_name = next(iter(list(class_name)))
            if package_name == "transformers":
                preformatted_task_data = self.show_transformers_tasks(class_name=class_name)
            elif package_name == "diffusers":
                code_name = get_code_names(class_name, package_name)
                preformatted_task_data = self.show_diffusers_tasks(code_name=code_name, class_name=class_name)
                preformatted_task_data.sort()
            elif package_name == "mflux":
                preformatted_task_data = ["Image", "Redux", "Kontext", "Depth", "Fill", "ConceptAttention", "ControlNet", "CavTon", "IC-Edit"]
            if preformatted_task_data:
                filtered_tasks = [task for task in preformatted_task_data for snip in snip_words if snip not in task]
                return package_name, class_name, filtered_tasks


def main(mir_db: MIRDatabase = None):
    """Parse arguments to feed to dict header reader"""
    import argparse
    import asyncio
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scrape the task classes from currently installed libraries and attach them to an existing MIR database.\nOffline function.",
            usage="mir-tasks",
            epilog="Should be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    from nnll.mir.automata import assimilate

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = AutoPkg()
    task_tuple = asyncio.run(auto_pkg.detect_tasks(mir_db))
    # nfo(task_tuple)

    assimilate(mir_db, [task for task in task_tuple])

    mir_db.write_to_disk()


def pipe(mir_db: MIRDatabase = None):
    import argparse
    import asyncio
    from sys import modules as sys_modules
    # from pprint import pprint

    if "pytest" not in sys_modules:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Scrape the task classes from currently installed libraries and attach them to an existing MIR database.\nOffline function.",
            usage="mir-tasks",
            epilog="Should be used after `mir-maid`.\n\nOutput:\n    INFO     ('Wrote #### lines to MIR database file.',)",
        )
        parser.parse_args()

    from nnll.mir.automata import assimilate

    if not mir_db:
        mir_db = MIRDatabase()

    auto_pkg = AutoPkg()
    pipe_tuple = asyncio.run(auto_pkg.detect_pipes(mir_db))
    # pprint(pipe_tuple)
    assimilate(mir_db, [pipe for pipe in pipe_tuple])
    mir_db.write_to_disk()


if __name__ == "__main__":
    pipe()
