# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from typing import List, Dict, Optional
from nnll.mir.json_cache import JSONCache, TEMPLATE_PATH_NAMED
from nnll.configure.constants import PARAMETERS_SUFFIX, BREAKING_SUFFIX

TEMPLATE_CONFIG = JSONCache(TEMPLATE_PATH_NAMED)


def make_mir_tag(repo_title: str, decoder=False, data: dict = None) -> List[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated"""
    import re

    root = "decoder" if decoder else "*"
    repo_title = repo_title.split(":latest")[0]
    repo_title = repo_title.split(":Q")[0]
    repo_title = repo_title.split(r"/")[-1].lower()
    pattern = r"^.*[v]?(\d{1}+\.\d).*"
    match = re.findall(pattern, repo_title)
    if match:
        if next(iter(match)):
            repo_title = repo_title.replace(next(iter(match))[-1], "")
    parts = repo_title.replace(".", "").split("-")
    if len(parts) == 1:
        parts = repo_title.split("_")

    clean_parts = [re.sub(PARAMETERS_SUFFIX, "", segment.lower()) for segment in parts]
    cleaned_string = "-".join([x for x in clean_parts if x])
    suffix_match = re.findall(BREAKING_SUFFIX, cleaned_string)  # Check for breaking suffixes first
    if suffix_match:
        suffix = next(iter(suffix for suffix in suffix_match[0] if suffix))
        cleaned_string = re.sub(suffix.lower(), "-", cleaned_string).rstrip("-,")
    else:
        suffix = root
    cleaned_string = re.sub(r"[._]+", "-", cleaned_string.lower()).strip("-_")

    return (cleaned_string, suffix)


def class_to_mir_tag(mir_db: Dict[str, str], id_tag: str) -> Optional[str]:
    """Converts a class identifier to its corresponding MIR tag.\n
    :param mir_db: A dictionary mapping series-compatibility pairs to their respective data.
    :param id_tag: The class identifier to convert.
    :return: An optional list containing the series and compatibility if found, otherwise None."""
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    @TEMPLATE_CONFIG.decorator
    def _read_data(data: Optional[Dict[str, str]] = None):
        return data["arch"]["transformer"]

    template_data = list(_read_data())
    for series, compatibility_data in mir_db.database.items():
        if any([template for template in template_data if template in series.split(".")[1]]):
            for compatibility, field_data in compatibility_data.items():
                if id_tag == series.split(".")[2]:
                    return [series, compatibility]

                class_name = MODEL_MAPPING_NAMES.get(id_tag, False)
                if not class_name:
                    return None
                pkg_data = field_data.get("pkg")
                if pkg_data:
                    for index_num, pkg_type_data in pkg_data.items():
                        maybe_class = pkg_type_data.get("transformers")
                        if maybe_class == class_name:
                            return [series, compatibility]
    return None


def mir_package(mir_db: Dict[str, str]):
    mir_db.find_path()
    pass
