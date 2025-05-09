# # // SPDX-License-Identifier: MPL-2.0

# # // d a r k s h a p e s

# import os
# import unittest
# from tempfile import TemporaryDirectory
# from pathlib import Path
# import re
# from nnll_42 import populate_module_index


# class TestPopulateModuleIndex(unittest.TestCase):
#     def test_populate_module_index(self):
#         with TemporaryDirectory() as tmpdir:
#             os.makedirs(Path(tmpdir) / "1")
#             (Path(tmpdir) / "1" / "indicator.txt").write_text("def func1(): pass\nclass Class1: pass")
#             os.makedirs(Path(tmpdir) / "2")
#             (Path(tmpdir) / "2" / "indicator.txt").write_text("def func2(): pass")

#             active_directories = [str(Path(tmpdir) / d) for d in ["1", "2"]]
#             result = populate_module_index(active_directories, ["indicator.txt"])

#             # regex to ensure alternating result matches
#             key = next(iter(result.keys()))
#             patterns = [r"Class1, func1", r"func1, Class1"]
#             pattern_match = next((m.group() for m in (re.search(p, key) for p in patterns) if m), "No match")
#             file_path = os.path.join(Path(tmpdir), "1", "indicator.txt")

#             expected = {
#                 f"[1 - {pattern_match}]: ({file_path},)[2 - func2]": f"({os.path.join(Path(tmpdir), '2', 'indicator.txt')})",
#             }
#             print(f"expected : {expected}")
#             print(f"result : {result}")
#             self.assertDictEqual(result, expected)


# if __name__ == "__main__":
#     unittest.main()
