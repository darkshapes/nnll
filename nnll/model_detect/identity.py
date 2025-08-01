# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable

import os


class ModelIdentity:
    def __init__(self):
        from nnll.metadata.model_tags import ReadModelTags
        from nnll.mir.maid import MIRDatabase

        self.mir_db = MIRDatabase()
        self.database = self.mir_db.database
        self.find_path = self.mir_db.find_path
        self.reader = ReadModelTags()

    async def label_model_class(self, base_model: str) -> str | None:
        """Attempts to identify and return a MIR tag for the given model class name\n
        :param base_model: The name or path of the model. If it contains a directory separator only the remainder will be considered.
        :return: A dictionary representing the  MIR tag if a match is found; otherwise, None."""
        if r"/" in base_model:
            base_model = os.path.basename(base_model)
        pkg_data = [base_model, base_model.lower(), base_model.title()]
        class_names = [*pkg_data]
        for data in pkg_data:
            data = f"{data.replace('-', '').replace('.', '').split(':')[0]}"
            class_suffixes = ["Pipeline", "Multimodal", "Text", "Model", "VLModel"]
            for segment in class_suffixes:
                data_suffixed = [f"{data}{segment}"]
            class_names.extend([data, *data_suffixed])
        for name in class_names:
            if mir_tag := self.find_path(field="pkg", target=name, sub_field="0"):
                return mir_tag

    async def label_model_layers(self, repo_id: str) -> list[str] | None:
        """Identifies and returns MIR tags associated with the layers of a model folder from disk cache.\n
        :param repo_id: The identifier of the model repository whose layers are to be analyzed.
        :return:Found MIR tags for each model found within a directory; or None"""

        from nnll.download.hub_cache import get_hub_path
        from nnll.integrity.hash_256 import hash_layers_or_files
        from nnll.monitor.file import dbug as nfo

        async def scan_folder_hashes(folder_path_named: str) -> list[list[str]]:
            nfo(f"{os.path.join(root, folder_path_named)}")
            hashes: dict[tuple[str, str]] = await hash_layers_or_files(os.path.join(root, folder_path_named), layer=True, b3=True, unsafe=False)
            # hashes should be a dictionary of {file_name}, {hex_value}
            if hashes:
                for file_name, hash_data in hashes.items():
                    mir_tag = self.find_path(field="layer_b3", target=hash_data)
                    file_path_named = os.path.join(root, folder_path_named, file_name)
                    if mir_tag:
                        mir_tags.append(mir_tag)
                    elif os.path.islink(file_path_named) and os.readlink(file_path_named):
                        mir_tag = self.find_path(field="file_256", target=os.path.basename(file_path_named))
                        if mir_tag:
                            mir_tags.append(mir_tag)
            return mir_tags

        mir_tags = []
        model_folder_named = await get_hub_path(repo_id)
        for root, folders, files in os.walk(model_folder_named):
            mir_tags = await scan_folder_hashes(root)

        return mir_tags

    async def label_model(self, repo_id: str, base_model: str | None, cue_type: str) -> list[str] | None:
        """Infer MIR tags for model based on repository ID, base model architecture, and related classes\n
        :param repo_id: The identifier for the repository containing the model.
        :param base_model: Optional; the base model name which can be used to infer the model's class.
        :param cue_type: Specifies the source of the provider, e.g., "HUB" for huggingface hub.
        :return: A list of found tags; if no matches are found, returns None"""

        from nnll.mir.tag import class_to_mir_tag

        if cue_type == "HUB":
            if mir_tags := await self.label_model_layers(repo_id):
                return mir_tags
        if mir_tag := self.find_path(field="repo", target=repo_id):
            return [mir_tag]
        if base_model:
            if mir_tag := class_to_mir_tag(self.mir_db, base_model.lower()):
                return [mir_tag]
            if mir_tag := await self.label_model_class(base_model):
                return [mir_tag]
        if mir_tag := await self.label_model_class(repo_id):
            return [mir_tag]

    async def get_cache_path(self, repo_data: Callable, file_name: str, match_attr: str = "file_path") -> str | None:
        """Returns the file path from a repository based on a query.\n
        :param repo: Repository object with revisions and files information.
        :param query: String to search for within the repository files.
        :param match_attr: Attribute to find the file in, defaults to "file_path".
        :return: The matched file path or None if no match found."""

        if hasattr(repo_data, "revisions") and repo_data.revisions:
            file_path = [getattr(info, match_attr, []) for info in next(iter(repo_data.revisions)).files if file_name in str(getattr(info, match_attr, []))]
            if file_path:
                return file_path[-1]
        return None
