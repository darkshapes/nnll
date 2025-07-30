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

    async def label_model_repo(self, repo_id: str) -> tuple[dict | None]:
        mir_tag = self.find_path(field="repo", target=repo_id)
        return [mir_tag]

    async def label_model_layers(self, repo_id: str) -> tuple[dict | None]:
        from nnll.download.hub_cache import get_hub_path
        from nnll.integrity.hash_256 import hash_layers_or_files

        async def scan_folder_hashes(folder_path_named: str) -> list[list[str]]:
            hashes = await hash_layers_or_files(os.path.join(root, folder_path_named))
            if hashes:
                print(hashes)
                for file_name, hash_data in hashes.items():
                    mir_tag = self.find_path(field="layer_b3", target=hash_data)
                    if mir_tag:
                        mir_tags.append(mir_tag)
            return mir_tags

        mir_tags = []
        model_folder_named = await get_hub_path(repo_id)
        for root, folders, files in os.walk(model_folder_named):
            mir_tags = await scan_folder_hashes(root)

        return mir_tags

    async def label_model_class(self, pkg_data: str) -> tuple[dict | None]:
        from nnll.mir.tag import class_to_mir_tag

        if mir_tag := class_to_mir_tag(self.mir_db, pkg_data.lower()):
            return [mir_tag]
        if r"/" in pkg_data:
            pkg_data = os.path.basename(pkg_data)
        pkg_data = [pkg_data, pkg_data.lower(), pkg_data.title()]
        class_names = [*pkg_data]
        for data in pkg_data:
            data = f"{data.replace('-', '').replace('.', '').split(':')[0]}"
            class_suffixes = ["Pipeline", "Multimodal", "Text", "Model"]
            for segment in class_suffixes:
                data_suffixed = [f"{data}{segment}"]
            class_names.extend([data, *data_suffixed])
        for name in class_names:
            if mir_tag := self.find_path(field="pkg", target=name, sub_field="0"):
                return [mir_tag]

    async def get_model_path(self, repo_data: Callable, query: str, match_attr: str | None = None, path_attr: str = "file_path"):
        """Returns the file path from a repository based on a query.\n
        :param repo: Repository object with revisions and files information.
        :param query: String to search for within the repository files.
        :param match_attr: Attribute to match the query against, defaults to path_attr.
        :param path_attr: Attribute containing the file path, defaults to "file_path".
        :return: The matched file path or None if no match found."""

        if not match_attr:
            match_attr = path_attr
        if hasattr(repo_data, "revisions") and repo_data.revisions:
            file_path = [getattr(info, path_attr, []) for info in next(iter(repo_data.revisions)).files if query in str(getattr(info, match_attr, []))]
            if file_path:
                return file_path[-1]
        return None

