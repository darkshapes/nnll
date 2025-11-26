# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from unittest.mock import MagicMock, patch
from nnll.chip_stats import ChipStats
from nnll.helpers import ensure_path
import pytest


@pytest.fixture
def chip_stats():
    return ChipStats(debug=True)


def test_write_stats(chip_stats):
    with (
        patch("psutil.virtual_memory") as mock_vm,
        patch("psutil.cpu_percent") as mock_cpu,
        patch("torch.cuda.mem_get_info") as mock_cuda,
        patch("torch.mps.driver_allocated_memory") as mock_mps,
        patch("os.path.exists") as mock_exists,
        patch("os.mkdir") as mock_mkdir,
        patch("nnll.read_tags.MetadataFileReader.read_header") as mock_read,
    ):
        mock_vm.return_value = MagicMock(total=10000000000)
        mock_cuda.return_value = (1000000000, 2000000000)
        mock_mps.return_value = 500000000
        mock_cpu.return_value = 10000000000
        mock_exists.return_value = False
        mock_mkdir.return_value = None
        mock_read.return_value = {"data": {"devices": {"cpu": 10000000000}}}
        file_name = ensure_path(os.path.join(os.path.dirname(os.getcwd()), "nnll", "tests"), file_name="chip_stats.json")
        print(file_name)
        chip_stats.debug = True
        chip_stats.write_stats(folder_path_named="./tests/")
        assert chip_stats.stats["data"]["devices"]["cpu"] == 10000000000
        assert os.path.isfile(file_name)
        os.remove(file_name)
