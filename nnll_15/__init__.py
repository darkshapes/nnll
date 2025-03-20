#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

"""Register model types"""

# pylint: disable=line-too-long, import-outside-toplevel

from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, computed_field
# import open_webui
# from package import response_panel


VALID_CONVERSIONS = ["text", "image", "music", "speech", "video", "upscale_image"]
OLLAMA_TASKS = {("image", "text"): ["mllama", "llava", "vllm"]}
LMS_TASKS = {("text", "text"): ["llm"], ("image", "text"): [True]}
HUB_TASKS = {
    ("text", "image"): ["image-generation", "image-text-to-text", "visual-question-answering"],
    ("text", "text"): ["chat", "conversational", "text-generation"],
    ("text", "video"): ["video generation"],
    ("speech", "text"): ["speech-translation", "speech-summarization", "automatic-speech-recognition"],
}


class RegistryEntry(BaseModel):
    """Validate Hub / Ollama / LMStudio model input"""

    size: int
    tags: list[str]
    library: str

    @computed_field
    @property
    def available_tasks(self) -> List[Tuple]:
        """Filter tag tasks into edge coordinates"""
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


def sort_dates_by_age(date_strings):
    """Parses datetime objects from strings and returns a list of dates sorted by age."""

    from datetime import datetime

    dates = []
    for date_string in date_strings:
        try:
            # Attempt to parse the datetime string
            date_object = datetime.strptime(date_string, "%Y, %m, %d, %H, %M, %S, %f")
            dates.append(date_object)
        except ValueError:
            print(f"Invalid date format: {date_string}")
            continue
    # Sort dates in ascending order (oldest to newest)
    return dates


def _extract_model_info(source: str, model_data: dict = None) -> Dict[str, Any]:
    """Helper function to extract common model information."""
    cache_dir = []
    cache_sizes = []
    model_tags = []
    timestamps = []
    models = []

    if source == "ollama":
        cache_dir = [f"ollama_chat/{model.model}" for model in model_data.models]
        cache_sizes = [model.size.real for model in model_data.models]
        model_tags = [[model.details.family] for model in model_data.models]
        timestamps = [[model.modified_at] for model in model_data.models]

    elif source == "hub":
        from huggingface_hub import repocard

        cache_dir = [obj.repo_id for obj in model_data.repos]
        cache_sizes = [obj.size_on_disk for obj in model_data.repos]
        metadata = [repocard.RepoCard.load(repo_name.repo_id) for repo_name in model_data.repos]
        repo_details = [obj.data for obj in metadata]
        timestamps = [obj.last_modified for obj in model_data.repos]

        for obj in repo_details:
            current_tag = []
            if hasattr(obj, "tags"):
                current_tag.extend([*obj.tags])
            if hasattr(obj, "pipeline_tag"):
                current_tag.append(obj.pipeline_tag)
            model_tags.append(current_tag if current_tag else ["unknown"])

    # elif source == "lms":
    #     import lmstudio as lms

    #     for model in model_data:
    #         if not hasattr(model, "model_key"):
    #             continue  # Skip models without a key
    #         name = model.model_key
    #         try:
    #             llm_model = lms.llm(name)
    #             cache_dir.append(f"lmstudio/{name}")
    #             cache_sizes.append(llm_model.get_info().size_bytes)
    #             tags = [llm_model.get_info().type]
    #             if llm_model.get_info().vision:
    #                 tags.append("vision")
    #             model_tags.append(tags)

    #         except (lms.LMStudioServerError, AttributeError):
    #             continue

    else:
        raise ValueError(f"Unsupported source: {source}")

    for model, size, tasks, timestamp in zip(cache_dir, cache_sizes, model_tags, timestamps):
        entry = RegistryEntry(size=size, tags=tasks, library=source)
        if getattr(entry, "available_tasks", []) != [("default_task:", None)]:
            models.append((model, entry, timestamp))

    models.sort(key=lambda x: x[2])  # Sort by timestamp
    return models


def from_ollama_cache() -> Dict[str, RegistryEntry]:
    """Retrieve models from ollama server."""
    from ollama import ListResponse, list as ollama_list

    model_data: ListResponse = ollama_list()
    return _extract_model_info("ollama", model_data)


def from_hf_hub_cache() -> Dict[str, RegistryEntry]:
    """Retrieve models from local huggingface hub cache."""
    from huggingface_hub import scan_cache_dir

    model_data = scan_cache_dir()
    return _extract_model_info("hub", model_data)


# def from_lms_cache() -> Dict[str, RegistryEntry]:
#     """Retrieve models from local lmstudio cache."""

#     import lmstudio as lms

#     lms_client = lms.get_default_client()
#     lms_client.api_host = "localhost:1143"
#     model_data = lms.list_downloaded_models()
#     return _extract_model_info("lms", model_data)
