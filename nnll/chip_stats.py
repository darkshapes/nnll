# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""System profiler utility"""

# pylint: disable=line-too-long
# pylint: disable=import-outside-toplevel

from functools import lru_cache
from typing import Any, Dict
from decimal import Decimal
from functools import cache
import os

from nnll import HOME_FOLDER_PATH
from nnll.console import nfo
from nnll.json_cache import JSONCache, CHIP_STATS_PATH_NAMED
from nnll.helpers import check_optional_import

CHIP_STATS_FILE = JSONCache(CHIP_STATS_PATH_NAMED)


class ChipStats:
    """GPU performance management and system profiler
    "write_stats" write static, launch time GPU configuration to stats file
    "get_stats" report current system utilization
    "read_stats" retrieve static, launch time environment configuration options from configuration file
    "_get_paths" internal helper to list app config paths
    "show_stats" display current and launch time stats in console
    """

    debug = False
    stats = 0

    def __init__(self, debug=False) -> None:
        self.write_stats()
        self.debug = debug

    @lru_cache
    def write_stats(self, folder_path_named: str = os.path.dirname(CHIP_STATS_PATH_NAMED), testing=debug) -> None:
        """Create a configuration file for low-level GPUspecifications\n
        :param folder_path_named: Path to the application configuration folder"""
        import multiprocessing as mp
        import os
        import platform

        import psutil
        import torch

        from nnll.init_gpu import set_torch_device

        torch.set_num_threads(1)

        mp.set_start_method("spawn", force=True)
        device = set_torch_device().type
        torch.set_num_threads(1)
        stats = dict()
        stats.setdefault("data", dict())
        stats["data"].setdefault("devices", dict())
        stats["data"].setdefault("torch", dict())
        stats["data"]["torch"].setdefault("dynamo", platform.system().lower() == "linux")
        if "cuda" in device:
            stats["data"]["devices"].setdefault("cuda", torch.cuda.mem_get_info()[1])
            stats["data"]["torch"].setdefault("flash_attention", torch.backends.cuda.flash_sdp_enabled() if platform.system().lower() == "linux" else False)
            stats["data"]["torch"].setdefault("allow_tf32", False)  # high-end datacenter gpus only
            stats["data"]["torch"].setdefault("xformers", torch.backends.cuda.mem_efficient_sdp_enabled())
            if "True" in [stats["data"]["torch"].get("xformers"), stats["data"].get("flash_attention")]:
                stats["data"]["torch"]["attention_slicing"] = False
            if torch.cuda.get_device_capability() >= (12, 0):
                stats["data"]["torch"].setdefault("triton", check_optional_import("triton"))
                stats["data"]["torch"].setdefault("sageattn", check_optional_import("sageattention"))
        if "mps" in device:
            if torch.backends.mps.is_available() & torch.backends.mps.is_built():
                # patches async issues with torch and MacOS
                mp.set_start_method("fork", force=True)
                stats["data"]["devices"].setdefault("mps", torch.mps.driver_allocated_memory())
                stats["data"]["torch"].setdefault("attention_slicing", False)
                if testing:
                    stats["data"]["torch"].setdefault("mps_memory_fraction", 1.7)
                    torch.mps.set_per_process_memory_fraction(stats["data"]["torch"]["mps_memory_fraction"])
                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        if "xpu" in device:
            stats["data"]["devices"].setdefault("xps", torch.xpu.mem_get_info())  # mostly just placeholder
        if "mtia" in device:
            stats["data"]["devices"].setdefault("mtia", torch.mtia.memory_stats())  # also mostly just placeholder
        stats["data"]["devices"].setdefault("cpu", int(psutil.virtual_memory().total))
        stats["data"]["torch"].setdefault(
            "versions",
            {
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "xpu": torch.version.xpu,
                "torch_git": torch.version.git_version,
            },
        )
        self.stats = stats
        if not self.debug and not testing:
            # consider: set cpu floats fp32?
            if not os.path.exists(folder_path_named):
                os.mkdir(folder_path_named)
            write_paths = [folder_path_named, "."]
            file_name = "chip_stats.json"
            for folder_path in write_paths:
                try:
                    from nnll.helpers import ensure_path

                    ensure_path(folder_path, file_name)
                    CHIP_STATS_FILE.update_cache(stats)
                except FileNotFoundError:
                    pass
                else:
                    break

    @lru_cache
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves current system metrics including CPU usage, RAM usage and disk usage. Caches results to optimize performance.\n
        :return: A dictionary of the system hardware state
            - "hostname" - network host name\n
            - "timestamp" - system clock\n
            - "cpu_%" - cpu utilization\n
            - "dram_%" - cpu utilization percentage\n
            - "dram_used" - allocated cpu memory\n
            - "dram_total" - all cpu memory\n
            - "disk_%" - disk utilization percentage\n
            - "disk_used_%" - allocated disk space for current task\n
            - "disk_total" - all disk space\n
            - "chip_stats" - static information from launch"""
        from datetime import datetime
        from socket import gethostname

        import psutil

        disk = psutil.disk_usage("/")
        ram = psutil.virtual_memory()
        chip_stats = self.read_stats()
        data = {
            "timestamp": datetime.now().strftime("%YY-%dd-%mm %HH:%MM:%Ss"),
            "cpu_%": Decimal(str(psutil.cpu_percent(interval=1))),
            "dram_%": psutil.virtual_memory().percent,
            "dram_used_%": Decimal(str(round(ram.used / (1024**3), 2))),
            "disk_%": round(disk.percent, 2),
            "disk_used_%": round(disk.used / (1024**3), 2),
            "dram_total_gb": Decimal(str(round(ram.total / (1024**3), 2))),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "hostname": gethostname(),
            "chip_stats": chip_stats,
            "paths": self._get_paths(),
        }
        return data

    @lru_cache
    def read_stats(self, folder_path_named: str = HOME_FOLDER_PATH) -> Dict[str, Any]:
        """Retrieve static, launch time environment configuration options from configuration file\n
        :param folder_path_named: Path to the application configuration folder
        :return: A mapping of the discovered flags\n
            - "attention_slicing - memory management\n
            - "devices" - available processors\n
            - "dynamo" - pipe compilation\n
            - "flash_attention" - additional memory package\n
            - "memory_fraction" - memory allocation\n
            - "tf32" tf32 format toggle\n
            - "xformers" - legacy memory management
            - "triton" - legacy memory management
            - "sageattn" - legacy memory management

            - "torch" - torch version
        """

        import os

        if not self.debug:
            from nnll.read_tags import MetadataFileReader

            reader = MetadataFileReader()
            if not self.stats:
                write_paths = [folder_path_named, "."]
                for folder_path in write_paths:
                    try:
                        self.stats = reader.read_header(file_path_named=os.path.join(folder_path, "chip_stats.json"))
                    except FileNotFoundError:
                        pass
                    else:
                        break
        stats = self.stats.get("data")
        chip_stats = {
            "attention_slicing": stats["torch"].get("attention_slicing", 0),
            "devices": stats.get("devices", 0),
            "dynamo": stats["torch"].get("dynamo", 0),
            "flash_attention": stats["torch"].get("flash_attention", 0),
            "memory_fraction": stats["torch"].get("mps_memory_fraction", 0),
            "tf32": stats["torch"].get("allow_tf32", 0),
            "xformers": stats["torch"].get("xformers", 0),
            "torch": stats["torch"].get("version"),
        }
        return chip_stats

    def _get_paths(self):
        from nnll import LOG_FOLDER_PATH, USER_PATH_NAMED
        from nnll.json_cache import VARIABLE_NAMES

        return {
            "home_folder": HOME_FOLDER_PATH,
            "app_settings": VARIABLE_NAMES,
            "app_data": USER_PATH_NAMED,
            "log_path": LOG_FOLDER_PATH,
        }

    async def show_stats(self, and_return: bool = True) -> dict[str, int | str | float | Decimal]:
        """System specifications for current and launch statistics\n
        :return: A dictionary of the system hardware state
            - "timestamp" - system clock\n
            - "cpu_%" - cpu utilization\n
            - "dram_%" - cpu utilization percentage\n
            - "dram_used" - allocated cpu memory\n
            - "dram_total" - all cpu memory\n
            - "disk_%" - disk utilization percentage\n
            - "disk_used_%" - allocated disk space for current task\n
            - "disk_total" - all disk space\n
            - "chip_stats" - static information from launch\n
            - "paths" - paths to the application configuration folder"""
        import os
        from pathlib import Path

        stats = self.get_stats()
        user_name = os.path.basename(Path.home())
        paths_to_strip = stats.copy()
        stats["paths"] = {name: path.replace(user_name, "____") for name, path in paths_to_strip["paths"].items() if isinstance(path, str)}
        nfo(stats)
        if and_return:
            return stats


def make_chip_stats(folder_path_named: str = HOME_FOLDER_PATH) -> Dict[str, Any]:
    """Create a system profile of important hardware and firmware settings on launch\n
    :param folder_path_named: Path to the application configuration folder
    :return: A mapping of parameters for retrieval
            - "home_folder" - home folder\n
            - "app_settings" - application settings\n
            - "app_data" - application data\n
            - "log_path" - log path"""

    @cache
    def _init_stats():
        stats = ChipStats()
        stats = stats.write_stats(folder_path_named)


CHIP_STATS = make_chip_stats()


def main():
    import asyncio

    chip_stats = ChipStats(debug=True)
    asyncio.run(chip_stats.show_stats())
    return nfo("Done.")


if __name__ == "__main__":
    main()
