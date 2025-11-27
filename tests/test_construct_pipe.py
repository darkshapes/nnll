# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


import unittest
from unittest.mock import patch
from nnll.tensor_pipe.construct_pipe import ConstructPipeline  # , pipe_call
from enum import Enum
import pytest

# todo - mock MIR db entry


class FakeType(Enum):
    """Model Provider constants\n"""

    CUDA: tuple = (True, "diffusers")
    MPS: tuple = (True, "diffusers")
    XPS: tuple = (False, "diffusers")


def async_pipe_call(func):
    async def wrapper(*args, **kwargs):
        # Strip None values from positional args (preserve order)
        filtered_args = tuple(a for a in args if a is not None)
        return await func(*filtered_args, **kwargs)

    return wrapper


arch_data = {
    "modules": {
        0: {
            "diffusers": "StableDiffusionXLPipeline",
            "generation": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
        }
    },
    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
    "repo": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "weight_map": "weight_maps/model.unet.stable-diffusion-xl:base.json",
}


class MockEntry:
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    modules = {
        0: {
            "diffusers": "StableDiffusionXLPipeline",
            "generation": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
        },
    }


class MockPrecision:
    database = {"ops.precision.float": {"F16": {"pkg": {"0": {"torch": {"float16": {"variant": "fp16"}}}}}}}


class MockKolors:
    model = "Kwai-Kolors/Kolors-diffusers"
    modules = {
        0: {
            "diffusers": "KolorsPipeline",
            "precision": "ops.precision.float.F16",
            "generation": {"negative_prompt": "", "guidance_scale": 5.0, "num_inference_steps": 50, "width": 1024, "height": 1024},
        }
    }


class MockWan:
    model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    modules = {
        0: {
            "diffusers": "WanTextToVideoPipeline",
        },
    }


class MockType(Enum):
    DIFFUSERS: tuple = (True, "DIFFUSERS", [])


@pytest.mark.asyncio
async def test_pipe_call_preserves_function_signature():
    async def test_pipe_call_preserves_function_signature(self):
        """Test that the decorator properly passes arguments"""

        @async_pipe_call
        async def sample_func(a, b=2, c=None):
            return (a, b, c)

        # Call with different argument patterns
        assert await sample_func(1) == (1, 2, None)
        assert await sample_func(1, 3, None) == (1, 3, None)
        assert await sample_func(1, c=4) == (1, 2, 4)
        assert await sample_func(1, 3, 4) == (1, 3, 4)

    async def test_pipe_call_ignores_none_values(self):
        """Ensure pipe_call ignores None values in args."""

        @async_pipe_call
        async def sample_func(a, b=None, c=3):
            return a, b, c

        assert await sample_func(1, None, 5) == (1, None, 5)
        assert await sample_func(1, b=None, c=6) == (1, None, 6)


@patch("os.path.isfile", return_value=True)
@patch("diffusers.StableDiffusionXLPipeline.from_single_file")
@pytest.mark.asyncio
async def test_create_pipeline_from_single_file(mock_from_single_file, mock_isfile):
    """Test pipeline creation from a single file"""
    mock_from_single_file.return_value = "mock_pipe"
    pipeline = ConstructPipeline()

    pipe, repo, import_pkg, settings = await pipeline.create_pipeline(
        registry_entry=MockEntry,
        pkg_data=(0, {"diffusers": "StableDiffusionXLPipeline"}, FakeType.MPS),
        mir_db={},
    )
    mock_from_single_file.assert_called_once_with(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
    )
    assert pipe == "mock_pipe"
    assert repo == "stabilityai/stable-diffusion-xl-base-1.0"
    assert settings == {"denoising_end": 0.8, "num_inference_steps": 40, "output_type": "latent", "safety_checker": False}


@patch("os.path.isfile", return_value=False)
@patch("diffusers.StableDiffusionXLPipeline.from_pretrained")
@pytest.mark.asyncio
async def test_create_pipeline_from_pretrained(mock_from_pretrained, mock_isfile):
    """Test pipeline creation from a pre-trained model"""
    mock_from_pretrained.return_value = "mock_pipe"

    pipeline = ConstructPipeline()
    pipe, repo, import_pkg, settings = await pipeline.create_pipeline(
        registry_entry=MockEntry,
        pkg_data=(0, {"diffusers": "StableDiffusionXLPipeline"}, FakeType.MPS),
        mir_db={},
    )

    mock_from_pretrained.assert_called_once_with(
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,
    )


@patch("os.path.isfile", return_value=False)
@patch("diffusers.KolorsPipeline.from_pretrained")
@pytest.mark.asyncio
async def test_create_kolors_pipe(mock_from_pretrained, mock_isfile):
    """Test pipeline creation from a pre-trained model"""
    # from nnll.mir.maid import MIRDatabase

    mock_from_pretrained.return_value = "mock_pipe"
    # mir_db = MIRDatabase()
    pipeline = ConstructPipeline()
    pipe, repo, import_pkg, settings = await pipeline.create_pipeline(
        registry_entry=MockKolors,
        pkg_data=(0, {"diffusers": "KolorsPipeline"}, FakeType.CUDA),
        mir_db=MockPrecision,
    )
    import torch

    mock_from_pretrained.assert_called_once_with("Kwai-Kolors/Kolors-diffusers", use_safetensors=True, torch_dtype=torch.float16, variant="fp16")


@patch("os.path.isfile", return_value=False)
@patch("diffusers.WanPipeline.from_pretrained")
@pytest.mark.asyncio
async def test_create_wan_pipe(mock_from_pretrained, mock_isfile):
    """Test pipeline creation from a pre-trained model"""
    mock_from_pretrained.return_value = "mock_pipe"

    pipeline = ConstructPipeline()

    pipe, repo, import_pkg, settings = await pipeline.create_pipeline(
        registry_entry=MockWan,
        pkg_data=("0", {"diffusers": "WanPipeline"}, FakeType.CUDA),
        mir_db={},
    )

    mock_from_pretrained.assert_called_once_with(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        use_safetensors=True,
    )


if __name__ == "__main__":
    # import asyncio
    # unittest.main()
    import pytest

    pytest.main(["-vv", __file__])
