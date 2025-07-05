import os
from typing import List
from nnll.mir.json_cache import JSONCache, VERSIONS_PATH_NAMED

VERSIONS_FILE = JSONCache(VERSIONS_PATH_NAMED)


def make_mir_tag(repo_title: str, decoder=False) -> List[str]:
    """Create a mir label from a repo path\n
    :param mir_prefix: Known period-separated prefix and model type
    :param repo_path: Typical remote source repo path, A URL without domain
    :return: The assembled mir tag with compatibility pre-separated
    """
    import re

    root = "decoder" if decoder else "*"

    parameters = r"[.-]?\d{1,4}[BbMmKk]|-tiny|-large|-medium|-base"
    parts = [segment for segment in re.split(parameters, repo_title) if segment]
    parts = "".join(parts[:1])
    version_pattern = r"([vV]?\d{1,2})(?:[-.]\d+)*"
    parts = [segment for segment in re.split(version_pattern, parts) if segment]
    parts = "".join(parts)
    exact = r".*(?:-)(prior)$|.*(?:-)(diffusers)$|.*(\d{3,4}px{1})"
    suffix = [tail for tail in re.split(exact, parts) if tail]
    parts = os.path.basename(parts).lower().replace("_", "-").replace(".", "-").replace("*", "")
    if len(suffix) > 1:
        return [parts, suffix]
    return [parts, root]
