# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import Callable, Awaitable
from functools import lru_cache
import os


class ModelIdentity:
    def __init__(self):
        from nnll.metadata.model_tags import ReadModelTags
        from nnll.mir.maid import MIRDatabase

        self.mir_db = MIRDatabase()
        self.database = self.mir_db.database
        self.find_tag = self.mir_db.find_tag
        self.reader = ReadModelTags()

    @lru_cache
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
            if mir_tag := self.find_tag(field="pkg", target=name, sub_field="0"):
                return mir_tag

    @lru_cache
    async def label_model_layers(self, repo_id: str, cue_type: str, repo_obj: Callable | None = None) -> list[str] | None:
        """Identifies and returns MIR tags associated with the layers of a model folder from disk cache.\n
        :param repo_id: The identifier of the model repository whose layers are to be analyzed.
        :return:Found MIR tags for each model found within a directory; or None"""

        from nnll.integrity.hash_256 import hash_layers_or_files
        from nnll.monitor.file import dbug as nfo

        @lru_cache
        async def scan_folder_hashes(folder_path_named: str) -> list[list[str]]:
            nfo(f"{os.path.join(root, folder_path_named)}")
            hashes: dict[tuple[str, str]] = await hash_layers_or_files(path_named=os.path.join(root, folder_path_named), layer=True, b3=True, unsafe=False)
            if hashes:
                for file_name, hex_value in hashes.items():
                    file_path_named = os.path.join(root, folder_path_named, file_name)
                    mir_tag = self.find_tag(field="layer_b3", target=hex_value)
                    if mir_tag:
                        mir_tags.insert(0, mir_tag)
                    elif os.path.islink(file_path_named) and os.readlink(file_path_named):  # check huggingface symlinks
                        if mir_tag := self.find_tag(field="file_256", target=os.path.basename(file_path_named)):
                            mir_tags.insert(0, mir_tag)
            return mir_tags

        mir_tags = []
        model_path_named = await self.get_cache_path(file_name=repo_id, repo_obj=repo_obj)
        if os.path.isdir(model_path_named):
            for root, folders, files in os.walk(model_path_named):
                mir_tags = await scan_folder_hashes(root)
        else:
            hash_data = await hash_layers_or_files(path_named=model_path_named, layer=True, b3=True, unsafe=False)
            if mir_tag := self.find_tag(field="layer_b3", target=next(iter(hash_data))):
                return [mir_tag]
            sha_sum = os.path.basename(model_path_named).replace("sha256-", "")
            if mir_tag := self.find_tag(field="file_256", target=sha_sum):
                return [mir_tag]

        return mir_tags

    @lru_cache
    async def label_model(self, repo_id: str, base_model: str | None, cue_type: str, repo_obj: Callable | None = None) -> list[str] | list[list[str]] | None:
        """Infer MIR tags for model based on repository ID, base model architecture, and related classes\n
        :param repo_id: The identifier for the repository containing the model.
        :param base_model: Optional; the base model name which can be used to infer the model's class.
        :param cue_type: Specifies the source of the provider, e.g., "HUB" for huggingface hub.
        :return: A list of found tags; if no matches are found, returns None"""

        from nnll.mir.tag import class_to_mir_tag

        print(repo_id)
        match_order = {  # ordered by most likely to match
            "HUB": [
                lambda: self.label_model_layers(repo_id, cue_type, repo_obj),
                lambda: self.find_tag(field="repo", target=repo_id),
                lambda: class_to_mir_tag(self.mir_db, base_model) if base_model else None,
                lambda: self.label_model_class(repo_id),
            ],
            "OLLAMA": [
                lambda: class_to_mir_tag(self.mir_db, base_model) if base_model else None,
                lambda: self.label_model_class(repo_id),
                lambda: self.label_model_layers(repo_id, cue_type, repo_obj),
                lambda: self.find_tag(field="repo", target=repo_id),
            ],
        }

        for labeler in match_order[cue_type]:
            mir_tag = labeler()
            if isinstance(mir_tag, Awaitable):
                if mir_tag := await mir_tag:
                    return mir_tag
            if mir_tag:
                return [mir_tag]

    @lru_cache
    async def get_cache_path(self, file_name: str, repo_obj: Callable | None = None) -> str | None:
        """Returns the file path from a repository based on a query.\n
        :param repo: Repository object with revisions and files information.
        :param query: String to search for within the repository files.
        :param match_attr: Attribute to find the file in, defaults to "file_path".
        :return: The matched file path or None if no match found."""

        if file_name and not repo_obj:
            from nnll.download.hub_cache import get_hub_path

            return await get_hub_path(file_name)
        elif repo_obj and isinstance(repo_obj, tuple):
            blob_path = repo_obj[0].partition(file_name)[-1].strip("\n")
            return blob_path.partition("\n")[0][4:].strip()
        elif repo_obj and hasattr(repo_obj, "revisions") and repo_obj.revisions:
            match_attr = "file_path"
            if file_path := [
                getattr(info, match_attr, [])
                for info in next(iter(repo_obj.revisions)).files
                if file_name
                in str(
                    getattr(info, match_attr, []),
                )
            ]:
                return file_path[-1]
        return None
