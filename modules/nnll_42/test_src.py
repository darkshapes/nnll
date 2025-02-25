### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import re
from modules.nnll_42.src import populate_module_index


class TestPopulateModuleIndex(unittest.TestCase):
    def test_populate_module_index(self):
        with TemporaryDirectory() as tmpdir:
            os.makedirs(Path(tmpdir) / "1")
            (Path(tmpdir) / "1" / "indicator.txt").write_text("def func1(): pass\nclass Class1: pass")
            os.makedirs(Path(tmpdir) / "2")
            (Path(tmpdir) / "2" / "indicator.txt").write_text("def func2(): pass")

            active_directories = [str(Path(tmpdir) / d) for d in ["1", "2"]]
            result = populate_module_index(active_directories, "indicator.txt")

            # regex to ensure alternating result matches
            expected = {
                f"[1 - {next((m.group() for m in (re.search(p, next(iter(result.keys()))) for p in [r'Class1, func1', r'func1, Class1']) if m), 'No match')}]": f"({os.path.join(Path(tmpdir), '1', 'indicator.txt')})",
                "[2 - func2]": f"({os.path.join(Path(tmpdir), '2', 'indicator.txt')})",
            }
            self.assertDictEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
