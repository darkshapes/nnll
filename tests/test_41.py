### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
import os
from tempfile import TemporaryDirectory

from nnll_41 import trace_project_structure


class TestTraceFileStructure(unittest.TestCase):
    """Temp folder, write empty file"""

    def test_with_multiple_indicators(self):
        with TemporaryDirectory() as tmpdir:
            parent_dir_1 = "nnll_1"
            parent_dir_2 = "nnll_2"
            file_name = "indicator.txt"
            os.makedirs(os.path.join(tmpdir, parent_dir_1))
            os.makedirs(os.path.join(tmpdir, parent_dir_2))

            with open(os.path.join(tmpdir, parent_dir_1, file_name), "w", encoding="UTF-8") as _:
                pass

            with open(os.path.join(tmpdir, parent_dir_2, file_name), "w", encoding="UTF-8") as _:
                pass
            # print(tmpdir)
            result = trace_project_structure(tmpdir)
            expected = [tmpdir, os.path.join(tmpdir, parent_dir_1), os.path.join(tmpdir, parent_dir_2)]
            self.assertListEqual(result, sorted(expected))


if __name__ == "__main__":
    unittest.main()
