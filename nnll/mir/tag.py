# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from typing import List, Dict, Optional
from nnll.mir.json_cache import JSONCache, TEMPLATE_PATH_NAMED
from nnll.configure.constants import PARAMETERS_SUFFIX, VERSIONS_SUFFIX, BREAKING_SUFFIX

TEMPLATE_CONFIG = JSONCache(TEMPLATE_PATH_NAMED)


def make_mir_tag(repo_title: str, decoder=False, data: dict = None) -> List[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated"""
    import re

    root = "decoder" if decoder else "*"
    parts = [segment for segment in re.split(PARAMETERS_SUFFIX, repo_title) if segment]
    parts = "".join(parts[:1])
    parts = [segment for segment in re.split(VERSIONS_SUFFIX, parts) if segment]
    parts = "".join(parts)
    suffix = [tail for tail in re.split(BREAKING_SUFFIX, parts) if tail]
    parts = os.path.basename(parts).lower().replace("_", "-").replace(".", "-").replace("*", "")
    if len(suffix) > 1:
        return [parts, suffix]
    return [parts, root]


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
