# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

"""類發現和拆卸"""

# pylint:disable=protected-access

from typing import Callable, Dict, List, Optional, Union, Type
from nnll.monitor.file import dbuq
from nnll.monitor.console import nfo
from mir.mappers import root_class, show_tasks_for, stock_llm_data, make_callable, class_parent


def get_code_names(
    class_name: Optional[Union[str, Type]] = None,
    pkg_name: Optional[str] = "transformers",
    path_format: Optional[bool] = False,
) -> Union[List[str], str]:
    """Reveal code names for class names from Diffusers or Transformers\n
    :param class_name: To return only one class, defaults to None
    :param library: optional field for library, defaults to "transformers"
    :return: A list of all code names, or the one corresponding to the provided class"""

    package_map = {
        "diffusers": ("_import_structure", "diffusers.pipelines"),
        "transformers": ("MODEL_MAPPING_NAMES", "transformers.models.auto.modeling_auto"),
    }
    pkg_name = pkg_name.lower()
    MAPPING_NAMES = make_callable(*package_map[pkg_name])
    if class_name:
        if isinstance(class_name, Type):
            class_name = class_name.__name__
        code_name = next(iter(key for key, value in MAPPING_NAMES.items() if class_name in str(value)), "")
        return class_parent(code_name, pkg_name) if path_format else code_name.replace("_", "-")
    return list(MAPPING_NAMES)


def find_config_classes(parameter_filter: Optional[str] = None) -> List[str]:
    """Show all config classes transformers package\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""

    transformers_data = stock_llm_data()
    config_data = []
    for model_path in list(transformers_data.values()):
        config_class = model_path["config"][-1]
        if parameter_filter:
            segments = root_class(config_class, pkg_name="transformers")
            if parameter_filter in list(segments):
                config_data.append(config_class)
        else:
            config_data.append(config_class)
    return config_data


def show_addons_for(model_class: Union[Callable, str], pkg_name: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    """Strips <class> tags from module's base classes and extracts inherited class members.\n
    If `module` is a string, it requires the `library` argument to convert it into a callable.\n
    :param module: A module or string representing a module.
    :param library: Library name required if `module` is a string. Defaults to None.
    :returns: Mapping indices to class path segments, or None if invalid input."""

    if isinstance(model_class, str):
        if not pkg_name:
            nfo("Provide a library type argument to process strings")
            return None
        model_class = make_callable(model_class, pkg_name)
    signature = model_class.__bases__
    class_names = []
    for index, class_annotation in enumerate(signature):
        tag_stripped = str(class_annotation)[8:-2]
        module_segments = tag_stripped.split(".")
        class_names.append(module_segments)
    return class_names


def trace_classes(pipe_class: str, pkg_name: str) -> Dict[str, List[str]]:
    """Retrieve all compatible pipe forms\n
    NOTE: Mainly for Diffusers
    :param pipe_class: Origin pipe
    :param pkg_name: Dependency package
    :return: A dictionary of pipelines"""

    related_pipes = []
    code_name = get_code_names(pipe_class, pkg_name)
    if pkg_name == "diffusers":
        related_pipe_class_name = pipe_class
    else:
        related_pipe_class_name = None
    related_pipes: list[str] = show_tasks_for(code_name=code_name, class_name=related_pipe_class_name)
    # for i in range(len(auto_tasks)):
    #     auto_tasks.setdefault(i, revealed_tasks[i])
    parent_folder = class_parent(code_name, pkg_name)
    if pkg_name == "diffusers":
        pkg_folder = make_callable(parent_folder[0], ".".join(parent_folder))
    else:
        pkg_folder = make_callable("__init__", ".".join(parent_folder[:-1]))
    if hasattr(pkg_folder, "_import_structure"):
        related_pipes.extend(next(iter(x)) for x in pkg_folder._import_structure.values())
    related_pipes = set(related_pipes)
    related_pipes.update(tuple(x) for x in show_addons_for(model_class=pipe_class, pkg_name=pkg_name))
    return related_pipes


def seek_class_path(class_name: str, pkg_name: str) -> List[str]:
    # from nnll.monitor.file import dbuq
    from nnll.tensor_pipe.deconstructors import get_code_names, root_class

    pkg_name = pkg_name.lower()
    if pkg_name == "diffusers":
        parent_folder: List[str] = get_code_names(class_name=class_name, pkg_name=pkg_name, path_format=True)
        if not parent_folder or not parent_folder[-1].strip():
            # dbuq("Data not found for", " class_name = {class_name},pkg_name = {pkg_name},{parent_folder} = parent_folder")
            return None
    elif pkg_name == "transformers":
        module_path = root_class(class_name, "transformers").get("config")
        parent_folder = module_path[:3]
    return parent_folder


# def pull_weight_map(repo_id: str, arch: str) -> Dict[str, str]:
#     from nnll.download.hub_cache import download_hub_file

#     model_file = download_hub_file(
#         repo_id=f"{repo_id}/tree/main/{arch}",
#         source="huggingface",
#         file_name="diffusion_pytorch_model.safetensors.index.json",
#         local_dir=".tmp",
#     )
