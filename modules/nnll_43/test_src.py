### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import os
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import re

from modules.nnll_42.src import populate_module_index
from modules.nnll_43.src import write_toc_to_file


class TestWriteTocToFile(unittest.TestCase):
    def test_write_toc_to_file(self):
        with TemporaryDirectory() as tmpdir:
            os.makedirs(Path(tmpdir) / "1")
            (Path(tmpdir) / "1" / "indicator.txt").write_text("def func1(): pass\nclass Class1: pass")
            os.makedirs(Path(tmpdir) / "2")
            (Path(tmpdir) / "2" / "indicator.txt").write_text("def func2(): pass")

            active_directories = [str(Path(tmpdir) / d) for d in ["1", "2"]]
            module_index = populate_module_index(active_directories, "indicator.txt")

            write_toc_to_file(module_index, index_file_name="__init__.py", walk_path=tmpdir)
            with open(Path(tmpdir) / "__init__.py") as f:
                content = f.read()

            pattern = r"(?:Class1, func1|func1, Class1)"

            # Use re.sub to replace the matched pattern with a placeholder that accounts for either/or situation
            expected_content = (
                "### <!-- // /*  SPDX-License-Identifier: blessing) */ -->\n"
                '### <!-- // /*  d a r k s h a p e s */ -->\n## module table of contents\n\n"""\n'
                f"#### [1 - {next((m.group() for m in (re.search(p, content) for p in [r'Class1, func1', r'func1, Class1']) if m), 'No match')}]({{}})\n"
                "#### [2 - func2]({})\n"
                '"""\n'
            ).format(Path(tmpdir) / "1" / "indicator.txt", Path(tmpdir) / "2" / "indicator.txt")

            # Replace the regex pattern with the actual content using re.sub
            actual_content = re.sub(pattern, "replace", expected_content)

            # Define a function to check both possibilities
            def matches_any_order(content):
                return any(re.search(p, content) for p in [r"Class1, func1", r"func1, Class1"])

            self.assertTrue(matches_any_order(content))
            self.assertEqual(content, expected_content)


if __name__ == "__main__":
    unittest.main()
