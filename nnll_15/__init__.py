#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel

from typing import Dict, List, Tuple
from pydantic import BaseModel, computed_field

from nnll_01 import debug_message, debug_monitor

# import open_webui
# from package import response_panel


VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "upscale_image"]
OLLAMA_TASKS = {("image", "text"): ["mllama", "llava", "vllm"]}
LMS_TASKS = {("text", "text"): ["llm"], ("image", "text"): [True]}
HUB_TASKS = {
    ("image", "text"): ["image-generation", "image-text-to-text", "visual-question-answering"],
    ("text", "text"): ["chat", "conversational", "text-generation"],
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


@debug_monitor
def _extract_model_info(source: str, model_data: dict = None) -> RegistryEntry:
    """
    Helper function to extract common model information.\n
    Output stacked by newest model first for each conversion type.\n
    :param source: Origin of this data (eg: HuggingFace, Ollama, CivitAI, ModelScope)
    :param model_data: Metadata of the local cache library of `source`
    :return: A class object containing model metadata relevant to execution\n
    """
    cache_dir = []
    cache_sizes = []
    timestamp = []
    model_tags = []

    if source == "ollama":
        cache_dir = [f"ollama_chat/{model.model}" for model in model_data.models]
        cache_sizes = [model.size.real for model in model_data.models]
        timestamp = [int(model.modified_at.timestamp()) for model in model_data.models]
        model_tags = [[model.details.family] for model in model_data.models]

    elif source == "hub":
        from huggingface_hub import repocard

        cache_dir = [obj.repo_id for obj in model_data.repos]
        cache_sizes = [obj.size_on_disk for obj in model_data.repos]
        timestamp = [int(obj.last_modified) for obj in model_data.repos]

        metadata = [repocard.RepoCard.load(repo_name.repo_id) for repo_name in model_data.repos]
        repo_details = [obj.data for obj in metadata]
        for obj in repo_details:  # retrieve model types from repocard tags
            current_tag = []
            if hasattr(obj, "tags"):
                current_tag.extend([*obj.tags])
            if hasattr(obj, "pipeline_tag"):
                current_tag.append(obj.pipeline_tag)
            model_tags.append(current_tag if current_tag else ["unknown"])

    else:
        debug_message(f"Unsupported source: {source}")
        raise ValueError(f"Unsupported source: {source}")

    models = []
    for model, size, tasks, ts in zip(cache_dir, cache_sizes, model_tags, timestamp):
        entry = RegistryEntry(model=model, size=size, tags=tasks, library=source, timestamp=ts)
        if getattr(entry, "available_tasks", []) != [("default_task:", None)]:
            models.append(entry)

    debug_message(models)
    models.sort(key=lambda x: x.timestamp, reverse=True)
    return models


@debug_monitor
def from_ollama_cache() -> Dict[str, RegistryEntry]:
    """Retrieve models from ollama server."""
    from ollama import ListResponse, list as ollama_list

    model_data: ListResponse = ollama_list()
    return _extract_model_info("ollama", model_data)


@debug_monitor
def from_hf_hub_cache() -> Dict[str, RegistryEntry]:
    """Retrieve models from local huggingface hub cache."""
    from huggingface_hub import scan_cache_dir

    model_data = scan_cache_dir()
    return _extract_model_info("hub", model_data)
