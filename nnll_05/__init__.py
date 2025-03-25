#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
from nnll_60 import JSONCache, CONFIG_PATH_NAMED

mir_db = JSONCache(CONFIG_PATH_NAMED)


@mir_db.decorator
def hf_repo_to_mir_arch(known_repo: str, mir_data: dict = None) -> str:
    mir_arch = next(key for key, value in mir_data() if known_repo in value["repo"])
    return mir_arch
