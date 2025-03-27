#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
from nnll_60 import JSONCache, CONFIG_PATH_NAMED

mir_db = JSONCache(CONFIG_PATH_NAMED)


@mir_db.decorator
def generate_from_hf_repo(known_repo: str, mir_data: dict = {}) -> str:
    """
    Find MIR id from known repo name and run generation\n
    MIR data and call instructions autofilled by decorator\n
    :param known_repo: HuggingFace repo name
    :param mir_data: MIR URI reference file
    :return: `str` of the mir URI
    """

    mir_arch = next(key for key, value in mir_data() if known_repo in value["repo"])
    import_path = mir_data[mir_arch].get("constructor")


#    importlib import_path[0]
#    run import path
#    call method
#    return output
