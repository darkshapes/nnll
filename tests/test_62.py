### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest.mock import patch, MagicMock
from nnll_01 import nfo
from nnll_62 import ConstructPipeline  # , pipe_call

# todo - mock MIR db entry

arch_data = {
    "dep_alt": {"diffusers": ["DiffusionPipeline"]},
    "gen_kwargs": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
    "init_kwargs": {"use_safetensors": True},
    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
    "repo": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "weight_map": "weight_maps/model.unet.stable-diffusion-xl:base.json",
}


arch_data = {
    "dep_alt": {"diffusers": ["DiffusionPipeline"]},
    "gen_kwargs": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
    "init_kwargs": {"use_safetensors": True},
    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
    "repo": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "weight_map": "weight_maps/model.unet.stable-diffusion-xl:base.json",
}
lcm = {"dep_pkg": {"diffusers": ["LCMScheduler"]}}
lora_name = ["info.lora.lcm", "stable-diffusion-xl"]
lora_repo = "latent-consistency/lcm-lora-sdxl"

scheduler_kwargs = {"timestep_spacing": "trailing"}

init_kwargs = {"fuse": 1.0}
kwargs = {"scheduler_name": lcm, "scheduler_kwargs": scheduler_kwargs}


class TestPipeCallDecorator(unittest.TestCase):
    def test_pipe_call_preserves_function_signature(self):
        """Test that the decorator properly passes arguments"""

        # @pipe_call
        def sample_function(a, b=2, c=None):
            return (a, b, c)

        # Call with different argument patterns
        self.assertEqual(sample_function(1), (1, 2, None))
        self.assertEqual(sample_function(1, 3, None), (1, 3, None))
        self.assertEqual(sample_function(1, c=4), (1, 2, 4))
        self.assertEqual(sample_function(1, 3, 4), (1, 3, 4))

    def test_pipe_call_ignores_none_values(self):
        """Ensure pipe_call ignores None values in args."""

        # @pipe_call
        def sample_func(a, b=None, c=3):
            return a, b, c

        self.assertEqual(sample_func(1, None, 5), (1, None, 5))
        self.assertEqual(sample_func(1, b=None, c=6), (1, None, 6))


class TestConstructPipeline(unittest.TestCase):
    # async def setUp(self):
    #     self.pipeline = ConstructPipeline()

    def test_get_module(self):
        pipeline = ConstructPipeline()
        module = pipeline._get_module({"diffusers": ["StableDiffusionXLPipeline"]})
        from diffusers import StableDiffusionXLPipeline

        assert module == [StableDiffusionXLPipeline]

    def test_get_several_module(self):
        pipeline = ConstructPipeline()
        module_1, module_2 = pipeline._get_module({"diffusers": ["StableDiffusionXLPipeline"], "transformers": ["CLIPModel"]})
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPModel

        assert module_1 == StableDiffusionXLPipeline
        assert module_2 == CLIPModel

    def test_sub_cls_locs(self):
        pipeline = ConstructPipeline()
        module = pipeline._get_module({"diffusers": ["StableDiffusionXLPipeline"]})
        class_sig = pipeline._get_sub_cls_locs(module[-1])
        expected_output = {
            "vae": ["diffusers", "models", "autoencoders", "autoencoder_kl", "AutoencoderKL"],
            "text_encoder": ["transformers", "models", "clip", "modeling_clip", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "models", "clip", "modeling_clip", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "models", "clip", "tokenization_clip", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "models", "clip", "tokenization_clip", "CLIPTokenizer"],
            "unet": ["diffusers", "models", "unets", "unet_2d_condition", "UNet2DConditionModel"],
            "scheduler": ["KarrasDiffusionSchedulers"],
            "image_encoder": ["transformers", "models", "clip", "modeling_clip", "CLIPVisionModelWithProjection"],
            "feature_extractor": ["transformers", "models", "clip", "image_processing_clip", "CLIPImageProcessor"],
        }
        assert class_sig == expected_output

    def test_sub_cls_load(self):
        pipeline = ConstructPipeline()
        prev_output = {
            "vae": ["diffusers", "models", "autoencoders", "autoencoder_kl", "AutoencoderKL"],
            "text_encoder": ["transformers", "models", "clip", "modeling_clip", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "models", "clip", "modeling_clip", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "models", "clip", "tokenization_clip", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "models", "clip", "tokenization_clip", "CLIPTokenizer"],
            "unet": ["diffusers", "models", "unets", "unet_2d_condition", "UNet2DConditionModel"],
            "scheduler": ["KarrasDiffusionSchedulers"],
            "image_encoder": ["transformers", "models", "clip", "modeling_clip", "CLIPVisionModelWithProjection"],
            "feature_extractor": ["transformers", "models", "clip", "image_processing_clip", "CLIPImageProcessor"],
        }

        for _, package in prev_output.items():
            import_map = {package[0]: package[-1:]}
            print(package)
            print(import_map)
            # current_import = pipeline._get_module(import_map)
            # # from importlib import import_module
            # # # import diffusers

            # path = ".".join(package[1:])
            # path_alt = package[-1]
            # mod = package[0]
            # print(f"{path}.{mod}")
            # print(f"{path_alt}")
            # # expected_import = import_module(mod)
            # expected_import = __import__(".".join(package[0:-1]))
            # # __import__()

            # assert current_import == expected_import

    @patch("os.path.isfile", return_value=True)
    @patch("diffusers.StableDiffusionXLPipeline.from_single_file")
    def test_create_pipeline_from_single_file(self, mock_from_single_file, mock_isfile):
        """Test pipeline creation from a single file"""
        mock_from_single_file.return_value = "mock_pipe"
        pipeline = ConstructPipeline()

        pipe, repo, import_pkg, settings = pipeline.create_pipeline(
            arch_data=arch_data,
            init_modules={"dep_pkg": {"diffusers": ["StableDiffusionXLPipeline"]}},
        )
        mock_from_single_file.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
        self.assertEqual(pipe, "mock_pipe")
        self.assertEqual(repo, "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(
            settings,
            {"denoising_end": 0.8, "num_inference_steps": 40, "output_type": "latent", "safety_checker": False},
        )

    @patch("os.path.isfile", return_value=False)
    @patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
    def test_create_pipeline_from_pretrained(self, mock_from_pretrained, mock_isfile):
        """Test pipeline creation from a pre-trained model"""
        mock_from_pretrained.return_value = "mock_pipe"

        pipeline = ConstructPipeline()
        # with self.assertRaises(NotImplementedError):
        # with patch("huggingface_hub.hf_hub_download", autospec=True):
        #     with patch("huggingface_hub.snapshot_download", autospec=True):
        pipe, repo, import_pkg, settings = pipeline.create_pipeline(
            arch_data=arch_data,
            init_modules={"dep_pkg": {"diffusers": ["StableDiffusionXLPipeline"]}},
        )

        mock_from_pretrained.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)

    @patch("os.path.isfile", return_value=False)
    @patch("diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights")
    @patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
    @patch("os.path.basename", return_value="mock_adapter")
    def test_add_lora(self, mock_basename, mock_from_pretrained, mock_lora_weights, mock_isfile):
        """Test add_lora method in ConstructPipeline."""
        mock_pipe = MagicMock()
        mock_lora = MagicMock()
        mock_from_pretrained.return_value = mock_pipe
        mock_lora_weights.return_value = mock_lora
        pipeline = ConstructPipeline()
        # with self.assertRaises(NotImplementedError):
        # with patch("huggingface_hub.hf_hub_download", autospec=True):
        #     with patch("huggingface_hub.snapshot_download", autospec=True):

        with patch("diffusers.LCMScheduler") as mock_scheduler_class:
            mock_scheduler_instance = MagicMock()
            mock_scheduler_class.return_value = mock_scheduler_instance
            pipe = pipeline.add_lora(mock_pipe, lora_repo=lora_repo, init_kwargs=init_kwargs, scheduler_data=lcm, scheduler_kwargs=scheduler_kwargs)
            mock_scheduler_class.assert_called_once_with({"timestep_spacing": "trailing"})
            self.assertEqual(mock_pipe.scheduler, mock_scheduler_instance)
            pipe.load_lora_weights.assert_called_once_with("latent-consistency/lcm-lora-sdxl", adapter_name="mock_adapter")
            pipe.fuse_lora.assert_called_once_with(adapter_name="mock_adapter", lora_scale=1.0)

    @patch("os.path.isfile", return_value=False)
    @patch("diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights")
    @patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
    @patch("os.path.basename", return_value="mock_adapter")
    def test_add_lora_with_no_solver_end_to_end(self, mock_adapter, mock_from_pretrained, mock_lora_weights, mock_isfile):
        """Test add_lora when no solver or fuse is provided, complete run with mocked model"""
        mock_pipe = MagicMock()
        mock_lora = MagicMock()
        from nnll_60.mir_maid import MIRDatabase

        mir_db = MIRDatabase()
        data = mir_db.find_path("repo", "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA")
        lora_data = mir_db.database[data[0]][data[1]]
        init_data = mir_db.database[data[0]]["[init]"]
        mock_from_pretrained.return_value = mock_pipe
        mock_lora_weights.return_value = mock_lora
        mock_lora = MagicMock()
        pipeline = ConstructPipeline()
        pipe = pipeline.add_lora(mock_pipe, lora_repo=next(iter(lora_data["repo"])), init_kwargs=init_data)

        # self.assertEqual(repo, "stabilityai/stable-diffusion-xl-base-1.0")
        pipe.load_lora_weights.assert_called_once_with("SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep_LoRA", adapter_name="mock_adapter")
        nfo(kwargs)
        # self.assertEqual(kwargs, {"denoising_end": 0.8, "num_inference_steps": 40, "output_type": "latent", "safety_checker": False})


if __name__ == "__main__":
    # import asyncio
    unittest.main()
