
import unittest
import os
from tempfile import TemporaryDirectory

from modules.nnll_41.src import trace_file_structure

class TestTraceFileStructure(unittest.TestCase):
    def test_with_multiple_indicators(self):
        with TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, '1', 'sub1'))
            os.makedirs(os.path.join(tmpdir, '2'))
            open(os.path.join(tmpdir, '1', 'indicator.txt'), 'w').close()
            open(os.path.join(tmpdir, '2', 'indicator.txt'), 'w').close()
            result = trace_file_structure(tmpdir, 'indicator.txt')
            expected = [os.path.join(tmpdir, d) for d in ['1', '2']]
            self.assertListEqual(result, sorted(expected))

if __name__ == '__main__':
    unittest.main()
