### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
import os
from tempfile import TemporaryDirectory

from . import trace_file_structure


class TestTraceFileStructure(unittest.TestCase):
    """Temp folder, write empty file"""

    def test_with_multiple_indicators(self):
        with TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "folder1"))
            os.makedirs(os.path.join(tmpdir, "folder2"))

            with open(os.path.join(tmpdir, "folder1", "indicator.txt"), "w", encoding="UTF-8") as _:
                pass

            with open(os.path.join(tmpdir, "folder2", "indicator.txt"), "w", encoding="UTF-8") as _:
                pass
            print(tmpdir)
            result = trace_file_structure(tmpdir, r".*folder.*")
            expected = [os.path.join(tmpdir, d) for d in ["folder1", "folder2"]]
            self.assertListEqual(result, sorted(expected))


if __name__ == "__main__":
    unittest.main()
