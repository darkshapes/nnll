# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import torch
import set_dtype
import modules.lookup as look
from functools import wraps


def pipe_call(func):
    @wraps(func)
    def wrapper(
        model=None,
        torch_dtype=None,
        base_pipe=None,
        variant=None,
        use_safetensors=None,
        output_type=None,
        # vae=None,
        **kwargs,
    ):
        if model:
            kwargs.setdefault("model", model)
        if torch_dtype:
            kwargs.setdefault("torch_dtype", torch_dtype)
        if base_pipe:
            kwargs.setdefault("base_pipe", base_pipe)
        if variant:
            kwargs.setdefault("variant", variant)
        if use_safetensors:
            kwargs.setdefault("use_safetensors", use_safetensors)
        if output_type:
            kwargs.setdefault("output_type", output_type)
        return func(**kwargs)

    return wrapper


@pipe_call
def kolors_pipe(
    model: str = look.kolors,
    torch_dtype: torch.dtype = set_dtype.kolors,
) -> tuple:
    from diffusers import KolorsPipeline

    pipe = KolorsPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        variant=set_dtype.variant_kolors,
    )
    kwargs = {
        "negative_prompt": "",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
    }
    return pipe, kwargs


@pipe_call
def sdxl_base_pipe(
    model=look.sdxl_base_local,
    torch_dtype=set_dtype.sdxl_base,
    variant=set_dtype.variant_sdxl_base,
    use_safetensors=True,
) -> tuple:
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
    )

    kwargs = {
        "num_inference_steps": 40,
        "denoising_end": 0.8,
        "output_type": "latent",
    }

    # Combine with existing kwargs, giving priority to passed values
    return pipe, kwargs


@pipe_call
def sdxl_base_single_pipe(
    model=look.sdxl_base_local,
    torch_dtype=set_dtype.sdxl_base,
    variant=set_dtype.variant_sdxl_base,
    use_safetensors=True,
) -> tuple:
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_single_file(
        model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
    )

    kwargs = {
        "num_inference_steps": 40,
        "denoising_end": 0.8,
        "output_type": "latent",
    }

    return pipe, kwargs


@pipe_call
def autot2i_pipe(
    model: str,
    config: str,
    base_pipe=None,
    torch_dtype: torch.dtype = set_dtype.autopipe,
    variant: str = set_dtype.variant_autopipe,
    **kwargs,
) -> tuple:
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        config_name="model_index.json",
        pretrained_model_or_path=model,
        torch_dtype=torch_dtype,
        variant=variant,
    )
    if base_pipe is not None:
        pipe.text_encoder = base_pipe.text_encoder
        pipe.vae = base_pipe.vae

    return pipe, kwargs


@pipe_call
def sdxl_refiner_pipe(
    base_pipe=None,  # call with output_type: latent
    model: str = look.sdxl_refiner,
    torch_dtype: torch.dtype = set_dtype.sdxl_refiner,
    variant: str = set_dtype.variant_sdxl_refiner,
    use_safetensors: bool = True,
    **kwargs,
) -> tuple:
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
    )

    if base_pipe is not None:
        pipe.text_encoder_2 = base_pipe.text_encoder_2
        pipe.vae = base_pipe.vae

    kwargs = {
        "num_inference_steps": 40,
        "denoising_end": 0.8,
    }
    return pipe, kwargs


@pipe_call
def sdxl_i2i_pipe(
    model: str = look.sdxl_refiner,
    torch_dtype: torch.dtype = set_dtype.sdxl_refiner,
    variant: str = set_dtype.variant_sdxl_refiner,
    use_safetensors: bool = True,
) -> tuple:
    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        variant=variant,
        use_safetensors=use_safetensors,
    )

    kwargs = {}
    return pipe, kwargs


@pipe_call
def sigma_pipe(
    model: str = look.sigma,
    torch_dtype: torch.dtype = set_dtype.sigma,
) -> tuple:
    from diffusers import PixArtSigmaPipeline

    pipe = PixArtSigmaPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
    )

    kwargs = {}
    return pipe, kwargs


# @pipe_call
# def lumina2_pipe(
#     model: str = look.lumina,
#     torch_dtype: torch.dtype = set_dtype.lumina2,
#     **kwargs,
# ) -> tuple:
#     from diffusers importLumina2Text2ImgPipeline
#     pipe = Lumina2Text2ImgPipeline.from_pretrained(model, torch_dtype=torch_dtype)
#     defaults = {
#         "height": 1024,
#         "width": 1024,
#         "guidance_scale": 4.0,
#         "num_inference_steps": 50,
#         # "cfg_trunc_ratio"    : 0.25,
#         # "cfg_normalization"  : True,,
#     }
#     kwargs = kwargs | defaults
#     return pipe, kwargs
