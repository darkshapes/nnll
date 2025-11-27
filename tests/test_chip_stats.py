# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def chip_stats():
    from nnll.configure.chip_stats import ChipStats

    return ChipStats()


def test_write_stats(chip_stats):
    with (
        patch("psutil.virtual_memory") as mock_vm,
        patch("psutil.cpu_percent") as mock_cpu,
        patch("torch.cuda.mem_get_info") as mock_cuda,
        patch("torch.mps.driver_allocated_memory") as mock_mps,
        patch("os.path.exists") as mock_exists,
        patch("os.mkdir") as mock_mkdir,
        patch("nnll.metadata.read_tags.MetadataFileReader.read_header") as mock_read,
    ):
        mock_vm.return_value = MagicMock(total=10000000000)
        mock_cuda.return_value = (1000000000, 2000000000)
        mock_mps.return_value = 500000000
        mock_cpu.return_value = 10000000000
        mock_exists.return_value = False
        mock_mkdir.return_value = None
        mock_read.return_value = {"data": {"devices": {"cpu": 10000000000}}}
        file_name = os.path.join(os.getcwd(), "chip_stats.json")
        chip_stats.write_stats(folder_path_named=".tests/", testing=True)
        assert chip_stats.stats["data"]["devices"]["cpu"] == 10000000000
        assert os.path.isfile(file_name)
        os.remove(file_name)


def test_get_stats(chip_stats):
    with (
        patch("psutil.disk_usage") as mock_disk,
        patch("psutil.virtual_memory") as mock_vm,
        patch("psutil.cpu_percent") as mock_cpu,
        # patch("nnll.configure.chip_stats.ChipStats.get_stats") as mock_get_stats,
    ):
        mock_disk.return_value = MagicMock(used=50000000000, total=100000000000)
        mock_vm.return_value = MagicMock(used=2000000000, total=8000000000)
        mock_cpu.return_value = 25
        # mock_get_stats.return_value = {"attention_slicing": True}
        from decimal import Decimal

        stats = chip_stats.get_stats()

        assert stats["cpu_%"] == Decimal(str(25))
        assert stats["dram_used_%"] == Decimal(str(1.86))
        assert stats["disk_used_%"] == 46.57
        assert stats["chip_stats"]["attention_slicing"] is False
