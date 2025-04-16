#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel

from typing import Dict, List, Tuple
from pydantic import BaseModel, computed_field

from nnll_01 import debug_message as dbug, debug_monitor  # , info_message as nfo
from nnll_15.constants import VALID_CONVERSIONS, VALID_TASKS, LibType

# import open_webui
# from package import response_panel


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    model: str
    size: int
    tags: list[str]
    library: LibType
    timestamp: int
    # tokenizer: None

    @computed_field
    @property
    def available_tasks(self) -> List[Tuple]:
        """Filter tag tasks into edge coordinates for graphing"""
        import re

        default_task = None
        library_tasks = {}
        processed_tasks = []
        library_tasks = VALID_TASKS[self.library]
        if self.library == LibType.OLLAMA:
            default_task = ("text", "text")
        elif self.library == LibType.HUB:
            pattern = re.compile(r"(\w+)-to-(\w+)")
            for tag in self.tags:
                match = pattern.search(tag)
                if match and all(group in VALID_CONVERSIONS for group in match.groups()):
                    processed_tasks.append((match.group(1), match.group(2)))
        # placeholder for VLLM/LMSTUDIO Libraries
        for tag in self.tags:
            for (graph_src, graph_dest), tags in library_tasks.items():
                if tag in tags and (graph_src, graph_dest) not in processed_tasks:
                    processed_tasks.append((graph_src, graph_dest))
        if default_task and default_task not in processed_tasks:
            processed_tasks.append(default_task)
        return processed_tasks

    @classmethod
    @debug_monitor
    def from_model_data(cls) -> list[tuple[str]]:  # lib_type: LibType) model_data: tuple[frozenset[str]]
        """# todo - split into dependency-specific implementations
        Create RegistryEntry instances based on source\n
        Extract common model information and stack by newest model first for each conversion type.\n
        :param lib_type: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
        :return: A list of RegistryEntry objects containing model metadata relevant to execution\n
        """
        entries = []

        if LibType.OLLAMA:
            try:
                from ollama import ListResponse, list as ollama_list  # type: ignore
            except (ModuleNotFoundError, ImportError) as error_log:
                dbug(error_log)
                return
            model_data: ListResponse = ollama_list()
            for model in model_data.models:  # pylint:disable=no-member
                entry = cls(
                    model=f"ollama_chat/{model.model}",
                    size=model.size.real,
                    tags=[model.details.family],
                    library=LibType.OLLAMA,
                    timestamp=int(model.modified_at.timestamp()),
                )
                entries.append(entry)

        if LibType.HUB:
            try:
                from huggingface_hub import scan_cache_dir, repocard  # type: ignore
            except (ModuleNotFoundError, ImportError) as error_log:
                dbug(error_log)
                return

            model_data = scan_cache_dir()
            for repo in model_data.repos:
                try:
                    meta = repocard.RepoCard.load(repo.repo_id).data
                except ValueError as error_log:
                    dbug(error_log)
                    continue
                tags = []
                if hasattr(meta, "tags"):
                    tags.extend(meta.tags)
                if hasattr(meta, "pipeline_tag"):
                    tags.append(meta.pipeline_tag)
                if not tags:
                    tags = ["unknown"]
                entry = cls(model=repo.repo_id, size=repo.size_on_disk, tags=tags, library=LibType.HUB, timestamp=int(repo.last_modified))
                entries.append(entry)
        if LibType.LM_STUDIO:  # doesn't populate RegitryEntry yet
            try:
                from lmstudio import get_default_client, list_downloaded_models  # type: ignore

                lms_client = get_default_client()
                lms_client.api_host = "localhost:1143"
                model_data = list_downloaded_models()
            except (ModuleNotFoundError, ImportError) as error_log:
                dbug(error_log)

        if LibType.VLLM:  # placeholder
            try:
                import vllm  # type: ignore  # noqa: F401 #pylint:disable=unused-import
            except (ModuleNotFoundError, ImportError) as error_log:
                dbug(error_log)
        # else:
        #     nfo("Unsupported source")
        #     raise ValueError("Unsupported source")

        return sorted(entries, key=lambda x: x.timestamp, reverse=True)


@debug_monitor
def from_cache() -> Dict[str, RegistryEntry]:
    """
    Retrieve models from ollama server, local huggingface hub cache, !!! Incomplete! local lmstudio cache & vllm.
    我們不應該繼續為LMStudio編碼。 歡迎貢獻者來改進它。 LMStudio is not OSS, but contributions are welcome.
    """
    models = None
    models = RegistryEntry.from_model_data()
    return models
