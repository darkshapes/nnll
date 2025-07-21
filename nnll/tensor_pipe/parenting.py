# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional


def find_config_classes(parameter_filter: Optional[str] = None) -> List[str]:
    """Show all config classes transformers package\n
    :param from_match: Narrow the classes to only those with an exact key inside
    :return: A list of all Classes"""
    from nnll.mir.mappers import stock_llm_data
    from nnll.tensor_pipe.deconstructors import root_class

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


def show_tasks_for(code_name: str, class_name: Optional[str] = None) -> List[str]:
    """Return Diffusers/Transformers task pipes based on package-specific query\n
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
            for model_code, pipe_class_obj in task_map.items():
                if code_name in model_code:
                    alt_tasks.append(pipe_class_obj.__name__)

    elif code_name:
        from transformers.utils.fx import _generate_supported_model_class_names

        alt_tasks = _generate_supported_model_class_names(code_name)
    return alt_tasks


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


def class_parent(code_name: str, pkg_name: str) -> Optional[List[str]]:
    """Retrieve the folder path within a class. Only returns if it is a valid path in the system\n
    ### NOTE: in most cases `__module__` makes this redundant
    :param code_name: The internal name for the model in the third-party API.
    :param pkg_name: The API Package
    :return: A list corresponding to the path of the model, or None if not found
    :raises KeyError: for invalid pkg_name
    """
    import os
    from importlib import import_module

    pkg_paths = {
        "diffusers": "pipelines",
        "transformers": "models",
    }
    folder_name = code_name.replace("-", "_")
    pkg_name = pkg_name.lower()
    folder_path = pkg_paths[pkg_name]
    package_obj = import_module(pkg_name)
    folder_path_named = [folder_path, folder_name]
    pkg_folder = os.path.dirname(getattr(package_obj, "__file__"))
    # dbuq(os.path.exists(os.path.join(pkg_folder, *folder_path_named)))
    if os.path.exists(os.path.join(pkg_folder, *folder_path_named)) is True:
        import_path = [pkg_name]
        import_path.extend(folder_path_named)
        return import_path
