#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel

from typing import Dict, List, Tuple
from pydantic import BaseModel, computed_field

from nnll_01 import debug_message as dbug, debug_monitor, info_message as nfo
from nnll_05 import split_sequence_by
from nnll_60 import CONFIG_PATH_NAMED, JSONCache

# import open_webui
# from package import response_panel

mir_db = JSONCache(CONFIG_PATH_NAMED)

VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "3d", "upscale_image"]
OLLAMA_TASKS = {("image", "text"): ["mllama", "llava", "vllm"]}
LMS_TASKS = {("text", "text"): ["llm"], ("image", "text"): [True]}
HUB_TASKS = {
    ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
    ("text", "text"): ["chat", "conversational", "text-generation", "text2text-generation"],
    ("text", "video"): ["video generation"],
    ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
}


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    model: str
    size: int
    tags: list[str]
    library: str
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
        if self.library == "ollama":
            library_tasks = OLLAMA_TASKS
            default_task = ("text", "text")
        elif self.library == "lms":
            library_tasks = LMS_TASKS
        elif self.library == "hub":
            library_tasks = HUB_TASKS
            pattern = re.compile(r"(\w+)-to-(\w+)")
            for tag in self.tags:
                match = pattern.search(tag)
                if match and all(group in VALID_CONVERSIONS for group in match.groups()):
                    processed_tasks.append((match.group(1), match.group(2)))
        for tag in self.tags:
            for (graph_src, graph_dest), tags in library_tasks.items():
                if tag in tags and (graph_src, graph_dest) not in processed_tasks:
                    processed_tasks.append((graph_src, graph_dest))
        if default_task and default_task not in processed_tasks:
            processed_tasks.append(default_task)
        return processed_tasks

    @classmethod
    @debug_monitor
    def from_model_data(cls, source: str) -> list[tuple[str]]:  # model_data: tuple[frozenset[str]]
        """
        Create RegistryEntry instances based on source\n
        Extract common model information and stack by newest model first for each conversion type.\n
        :param source: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
        :param model_data: Metadata of the local cache library of `source`
        :return: A list of RegistryEntry objects containing model metadata relevant to execution\n
        """
        entries = []

        if source == "ollama":
            from ollama import ListResponse, list as ollama_list

            model_data: ListResponse = ollama_list()
            for model in model_data.models:  # pylint:disable=no-member
                entry = cls(
                    model=f"ollama_chat/{model.model}",
                    size=model.size.real,
                    tags=[model.details.family],
                    library=source,
                    timestamp=int(model.modified_at.timestamp()),
                )
                entries.append(entry)
        elif source == "hub":
            from huggingface_hub import scan_cache_dir

            model_data = scan_cache_dir()
            from huggingface_hub import repocard

            for repo in model_data.repos:
                meta = repocard.RepoCard.load(repo.repo_id).data
                tags = []
                if hasattr(meta, "tags"):
                    tags.extend(meta.tags)
                if hasattr(meta, "pipeline_tag"):
                    tags.append(meta.pipeline_tag)
                if not tags:
                    tags = ["unknown"]
                entry = cls(
                    model=repo.repo_id,
                    size=repo.size_on_disk,
                    tags=tags,
                    library=source,
                    timestamp=int(repo.last_modified),
                )
                entries.append(entry)
        elif source == "lms":
            try:
                import lmstudio as lms
            except ImportError as error_log:
                print("LMStudio not found")
                nfo(error_log)

            lms_client = lms.get_default_client()
            lms_client.api_host = "localhost:1143"
            model_data = lms.list_downloaded_models()
        else:
            dbug(f"Unsupported source: {source}")
            raise ValueError(f"Unsupported source: {source}")

        return sorted(entries, key=lambda x: x.timestamp, reverse=True)


@debug_monitor
def from_cache(ollama: bool = True, hub: bool = True, lms: bool = False) -> Dict[str, RegistryEntry]:
    """
    Retrieve models from ollama server, local huggingface hub cache, !!! Incomplete! local lmstudio cache.
    我們不應該繼續為LMStudio編碼。 歡迎貢獻者來改進它。 LMStudio is not OSS, but contributions are welcome.
    """
    if ollama:
        ollama_models = RegistryEntry.from_model_data("ollama")
    if hub:
        hub_models = RegistryEntry.from_model_data("hub")
    if lms:
        lms_models = RegistryEntry.from_model_data("lms")
    return {"ollama": ollama_models, "hub": hub_models, "lms": lms_models}
