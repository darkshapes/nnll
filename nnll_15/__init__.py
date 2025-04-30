#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel, protected-access, unsubscriptable-object

from typing import Dict, List, Tuple
from pydantic import BaseModel, computed_field

from nnll_01 import dbug, debug_monitor, nfo
from nnll_15.constants import LIBTYPE_CONFIG, VALID_CONVERSIONS, VALID_TASKS, LibType, has_api

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
    # api: None

    @computed_field
    @property
    def available_tasks(self) -> List[Tuple]:
        """Filter tag tasks into edge coordinates for graphing"""
        import re

        default_task = None
        library_tasks = {}
        processed_tasks = []
        library_tasks = VALID_TASKS[self.library]
        if self.library in [LibType.OLLAMA, LibType.LM_STUDIO, LibType.LLAMAFILE, LibType.CORTEX, LibType.VLLM]:
            default_task = ("text", "text") # usually these are txt gen libraries
        elif self.library == LibType.HUB: # pair tags from the hub such 'x-to-y' such as 'text-to-text' etc
            pattern = re.compile(r"(\w+)-to-(\w+)")
            for tag in self.tags:
                match = pattern.search(tag)
                if match and all(group in VALID_CONVERSIONS for group in match.groups()):
                    processed_tasks.append((match.group(1), match.group(2)))
        for tag in self.tags: # when pair-tagged elements are not available, potential to duplicate HUB tags here
            for (graph_src, graph_dest), tags in library_tasks.items():
                if tag in tags and (graph_src, graph_dest) not in processed_tasks:
                    processed_tasks.append((graph_src, graph_dest))
        if default_task and default_task not in processed_tasks:
            processed_tasks.append(default_task)
        return processed_tasks

    @classmethod
    def from_model_data(cls) -> list[tuple[str]]:  # lib_type: LibType) model_data: tuple[frozenset[str]]
        """Create RegistryEntry instances based on source\n
        Extract common model information and stack by newest model first for each conversion type.\n
        :param lib_type: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
        :return: A list of RegistryEntry objects containing model metadata relevant to execution\n"""
        entries = []

        @LIBTYPE_CONFIG.decorator
        def _read_data(data:dict =None):
            return data

        api_data = _read_data()

        if next(iter(LibType.OLLAMA.value)) and has_api("OLLAMA"): # check that server is still up!
            from ollama import ListResponse, list as ollama_list

            model_data: ListResponse = ollama_list()  # type: ignore
            for model in model_data.models:  # pylint:disable=no-member
                entry = cls(
                    model=f"{api_data[LibType.OLLAMA.value[1]].get('prefix')}{model.model}",
                    size=model.size.real,
                    tags=[model.details.family],
                    library=LibType.OLLAMA,
                    timestamp=int(model.modified_at.timestamp()),
                )
                entries.append(entry)
        if next(iter(LibType.HUB.value)) and has_api("HUB"):
            from huggingface_hub import scan_cache_dir, repocard, HFCacheInfo  # type: ignore

            model_data: HFCacheInfo = scan_cache_dir()
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

        if next(iter(LibType.CORTEX.value)) and has_api("CORTEX"):
            import requests
            from datetime import datetime

            response: requests.models.Request = requests.get(api_data["CORTEX"]["api_kwargs"]["api_base"], timeout=(3, 3))
            model: dict = response.json()
            for model_data in model["data"]:
                entry = cls(
                    model=f"{api_data[LibType.CORTEX.value[1]].get('prefix')}/{model_data.get('model')}",
                    size=model_data.get("size", 0),
                    tags=[str(model_data.get("modalities", "text"))],
                    library=LibType.CORTEX,
                    timestamp=int(datetime.timestamp(datetime.now())),  # no api for time data in cortex
                )
                entries.append(entry)

        if next(iter(LibType.LLAMAFILE.value)) and has_api("LLAMAFILE"):
            from openai import OpenAI

            model_data: OpenAI = OpenAI(base_url=api_data["LLAMAFILE"]["api_kwargs"]["api_base"], api_key="sk-no-key-required")
            for model in model_data.models.list().data:
                entry = cls(
                    model=f"{api_data[LibType.LLAMAFILE.value[1]].get('prefix')}/{model.id}",
                    size=0,
                    tags=["text"],
                    library=LibType.LLAMAFILE,
                    timestamp=int(model.created),  # no api for time data in cortex
                )
                entries.append(entry)

        if next(iter(LibType.VLLM.value)) and has_api("VLLM"):  # placeholder
            # import vllm

            model_data = OpenAI(base_url=api_data["VLLM"]["api_kwargs"]["api_base"], api_key=api_data["VLLM"]["api_kwargs"]["api_key"])
            for model in model_data.models.list().data:
                entry = cls(
                    model = f"{api_data[LibType.VLLM.value[1]].get('prefix')}{model['data'].get('id')}f",
                    size=0,
                    tags=["text"],
                    library=LibType.VLLM,
                    timestamp=int(model.created),  # no api for time data in cortex
                )
                entries.append(entry)

        if next(iter(LibType.LM_STUDIO.value)) and has_api("LM_STUDIO"):
            from lmstudio import list_downloaded_models  # pylint: disable=import-error, # type: ignore
            model_data = list_downloaded_models()
            for model in model_data:  # pylint:disable=no-member
                tags = []
                if hasattr(model._data, "vision"):
                    tags.extend("vision", model._data.vision)
                if hasattr(model._data, "trained_for_tool_use"):
                    tags.append(("tool", model._data.trained_for_tool_use))
                entry = cls(
                    model=f"{api_data[LibType.LM_STUDIO.value[1]].get('prefix')}{model.model_key}",
                    size=model._data.size_bytes,
                    tags=tags,
                    library=LibType.LM_STUDIO,
                    timestamp=int(model.modified_at.timestamp()),
                )
                entries.append(entry)
        # else:
        #     nfo("Unsupported source")
        #     raise ValueError("Unsupported source")
        nfo(f"entries {entries}")
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)


@debug_monitor
def from_cache() -> Dict[str, RegistryEntry]:
    """
    Retrieve models from ollama server, local huggingface hub cache, !!! Incomplete! local lmstudio cache & vllm.
    我們不應該繼續為LMStudio編碼。 歡迎貢獻者來改進它。 LMStudio is not OSS, but contributions are welcome.
    """
    models = None
    models = RegistryEntry.from_model_data()
    dbug(f"REG_ENTRIES {models}")
    return models
