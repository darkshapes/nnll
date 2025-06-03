### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

from typing import Any, Dict
from functools import cache
from nnll.configure import HOME_FOLDER_PATH


@cache
def make_chip_stats(folder_path_named: str = HOME_FOLDER_PATH) -> Dict[str, Any]:
    """Create a system profile of important hardware and firmware settings on launch\n
    :return: A mapping of parameters for retrieval
    """
    from nnll.configure.chip_stats import ChipStats

    stats = ChipStats()
    stats = stats.write_stats(HOME_FOLDER_PATH)
    return stats


CHIP_STATS = make_chip_stats()
