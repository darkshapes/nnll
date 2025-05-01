### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest.mock import patch, MagicMock

from nnll_62 import ConstructPipeline, pipe_call


class TestPipeCallDecorator(unittest.TestCase):
    def test_pipe_call_preserves_function_signature(self):
        """Test that the decorator properly passes arguments"""

        @pipe_call
        def sample_function(a, b=2, c=None):
            return (a, b, c)

        # Call with different argument patterns
        self.assertEqual(sample_function(1), (1, 2, None))
        self.assertEqual(sample_function(1, 3, None), (1, 3, None))
        self.assertEqual(sample_function(1, c=4), (1, 2, 4))
        self.assertEqual(sample_function(1, 3, 4), (1, 3, 4))

    def test_pipe_call_ignores_none_values(self):
        """Ensure pipe_call ignores None values in args."""

        @pipe_call
        def sample_func(a, b=None, c=3):
            return a, b, c

        self.assertEqual(sample_func(1, None, 5), (1, None, 5))
        self.assertEqual(sample_func(1, b=None, c=6), (1, None, 6))


class TestConstructPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = ConstructPipeline()

    @patch("os.path.isfile", return_value=True)
    @patch("diffusers.StableDiffusionXLPipeline.from_single_file")
    def test_create_pipeline_from_single_file(self, mock_from_single_file, mock_isfile):
        """Test pipeline creation from a single file"""
        mock_from_single_file.return_value = "mock_pipe"
        pipe, repo, settings = self.pipeline.create_pipeline("model.unet.stable-diffusion-xl:base")

        mock_from_single_file.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
        self.assertEqual(pipe, "mock_pipe")
        self.assertEqual(repo, "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(settings, {"denoising_end": 0.8, "num_inference_steps": 40, "output_type": "latent"})

    @patch("os.path.isfile", return_value=False)
    @patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
    def test_create_pipeline_from_pretrained(self, mock_from_pretrained, mock_isfile):
        """Test pipeline creation from a pre-trained model"""
        mock_from_pretrained.return_value = "mock_pipe"

        with self.assertRaises(NotImplementedError):
            self.pipeline.create_pipeline("model.unet.stable-diffusion-xl:base")

        mock_from_pretrained.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)

    @patch("os.path.basename", return_value="mock_adapter")
    def test_add_lora(self, mock_basename):
        """Test add_lora method in ConstructPipeline."""
        mock_pipe = MagicMock()

        with patch("diffusers.LCMScheduler") as mock_scheduler_class:
            mock_scheduler_instance = MagicMock()
            mock_scheduler_class.return_value = mock_scheduler_instance

            construct_pipeline = ConstructPipeline()
            pipe, repo, kwargs = construct_pipeline.add_lora("model.lora.lcm", "model.unet.stable-diffusion-xl:base", mock_pipe)

            # Validate pipe modification
            mock_scheduler_class.assert_called_once_with({"timestep_spacing": "trailing"})
            self.assertEqual(mock_pipe.scheduler, mock_scheduler_instance)
            pipe.load_lora_weights.assert_called_once_with("latent-consistency/lcm-lora-sdxl", adapter_name="mock_adapter")
            pipe.fuse_lora.assert_called_once_with(adapter_name="mock_adapter", lora_scale=1.0)
            self.assertEqual(repo, "latent-consistency/lcm-lora-sdxl")
            self.assertEqual(kwargs, {})

    def test_add_lora_with_no_solver(self):
        """Test add_lora when no solver is provided."""
        mock_pipe = MagicMock()

        construct_pipeline = ConstructPipeline()
        pipe, repo, kwargs = construct_pipeline.add_lora("model.lora.spo", "model.unet.stable-diffusion-xl:base", mock_pipe)

        self.assertEqual(repo, "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA")
        pipe.load_lora_weights.assert_called_once_with("SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA", adapter_name="SPO-SDXL_4k-p_10ep_LoRA")
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
