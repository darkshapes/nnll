### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# # pylint: disable=line-too-long

# pylint: disable=import-outside-toplevel

from functools import lru_cache
from typing import Any, Dict
from nnll.metadata.json_io import write_json_file
from nnll.configure import HOME_FOLDER_PATH


class ChipStats:
    """GPU performance management"""

    stats = 0

    @lru_cache
    def write_stats(self, folder_path_named: str = HOME_FOLDER_PATH, testing=False) -> None:
        """Create a configuration file for current system specifications\n
        :param folder_path_named: Path to the application configuration folder
        """
        from collections import defaultdict
        import platform
        import psutil
        import torch
        import multiprocessing as mp
        from nnll.configure.init_gpu import first_available

        mp.set_start_method("spawn", force=True)
        device = first_available(assign=False)
        stats = defaultdict(dict)
        stats["data"] = defaultdict(dict)
        stats["data"]["devices"] = defaultdict(dict)
        stats["data"]["dynamo"] = False if platform.system().lower() != "linux" else True
        if "cuda" in device:
            stats["data"]["devices"]["cuda"] = torch.cuda.mem_get_info()[1]
            stats["data"]["flash_attention"] = torch.backends.cuda.flash_sdp_enabled() if platform.system().lower() == "linux" else False
            stats["data"]["allow_tf32"] = False
            stats["data"]["xformers"] = torch.backends.cuda.mem_efficient_sdp_enabled()
            if "True" in [stats["data"].get("xformers"), stats["data"].get("flash_attention")]:
                stats["data"]["attention_slicing"] = False
        if "mps" in device:
            if torch.backends.mps.is_available() & torch.backends.mps.is_built():
                # patches async issues with torch and MacOS
                mp.set_start_method("fork", force=True)
                stats["data"]["devices"]["mps"] = torch.mps.driver_allocated_memory()
                stats["data"]["attention_slicing"] = True
                if testing:
                    stats["data"]["mps"]["memory_fraction"] = 1.7
                    torch.mps.set_per_process_memory_fraction(stats["data"]["mps"]["memory_fraction"])
                    import os

                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = 0.0
        if "xpu" in device:
            stats["data"]["devices"]["xps"] = torch.xpu.mem_get_info()  # mostly just placeholder
        if "mtia" in device:
            stats["data"]["devices"]["mtia"] = torch.mtia.memory_stats()  # also mostly just placeholder
        stats["data"]["devices"]["cpu"] = psutil.virtual_memory().total
        # consider: set cpu floats fp32?

        write_json_file(folder_path_named=folder_path_named, file_name="chip_stats.json", data=stats, mode="w")
        self.stats = stats

    @lru_cache
    def get_stats(self, folder_path_named: str = HOME_FOLDER_PATH) -> Dict[str, Any]:
        """Retrieve configuration options from configuration file\n
        :param folder_path_named: Path to the application configuration folder
        :return: A mapping of the discovered flags
        """

        from nnll.metadata.read_tags import MetadataFileReader
        import os

        reader = MetadataFileReader()
        if not self.stats:
            self.stats = reader.read_header(file_path_named=os.path.join(folder_path_named, "chip_stats.json"))
        stats = self.stats.get("data")
        chip_stats = {
            "attention_slicing": stats.get("attention_slicing", 0),
            "devices": stats.get("devices", 0),
            "dynamo": stats.get("dynamo", 0),
            "flash_attention": stats.get("flash_attention", 0),
            "memory_fraction": stats.get("set_per_process_memory_fraction", 0),
            "tf32": stats.get("allow_tf32", 0),
            "xformers": stats.get("xformers", 0),
        }
        return chip_stats
