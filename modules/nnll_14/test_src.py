
import unittest
from unittest.mock import patch, MagicMock
from sys import platform
import torch
from src import supported_backends

class TestSupportedBackends(unittest.TestCase):

    @patch("sys.platform", "Windows")
    def test_supported_backends_non_darwin(self):
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_cuda.is_built.return_value = True
        mock_cuda.device_count.return_value = 2

        mock_cpu = MagicMock()
        mock_cpu.is_available.return_value = True

        with patch.dict("torch.__dict__", {"cuda": mock_cuda, "cpu": mock_cpu}):
            devices = set(supported_backends())
            self.assertEqual(devices, {"cuda:0", "cuda:1", "cpu:0"})

    @patch("sys.platform", "Darwin")
    def test_supported_backends_darwin(self):

        mock_cpu = MagicMock()
        mock_cpu.is_available.return_value = True
        mock_cpu.is_built.return_value = True

        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        mock_mps.is_built.return_value = True

        with patch.dict("torch.__dict__", {"cpu": mock_cpu, "mps": mock_mps}):
            devices = set(supported_backends())
            self.assertEqual(devices, {"mps:0", "cpu:0"})

    @patch("sys.platform", "Linux")
    def test_unsupported_backend(self): # No torch mock
        with patch.dict("torch.__dict__", {None: None}):
            with self.assertRaises(RuntimeError) as context:
                list(supported_backends())
            self.assertIn("cuda is not an available device.", str(context.exception))

    @patch("sys.platform", "Linux")
    def test_unavailable_backend(self):
        # Mock torch.cuda to be available but not usable
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = False

        with patch.dict("torch.__dict__", {"cuda": mock_cuda}):
            #mock_system.return_value = "linux"
            with self.assertRaises(RuntimeError) as context:
                list(supported_backends())
            self.assertIn("cuda is not an available device.", str(context.exception))

    @patch("sys.platform", "Linux")
    def test_unconfigured_backend(self):
        # Mock torch.xpu to be available but not built
        mock_cuda = MagicMock()
        mock_cuda.is_available.return_value = True
        mock_cuda.is_built.return_value = False

        with patch.dict("torch.__dict__", {"cuda": mock_cuda}):
            with self.assertRaises(RuntimeError) as context:
                list(supported_backends())
            self.assertIn("cuda is an available but not a configured device.", str(context.exception))

if __name__ == "__main__":
    unittest.main()