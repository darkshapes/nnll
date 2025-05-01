### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from nnll_01 import debug_monitor


@debug_monitor
def add_hi_diffusion(pipe, kwargs):
    """Apply support for up to 4096 generation without upscaling
    compatibility: stable-diffusion-xl, stable-diffusion, stable-diffusion-2
    """
    from hidiffusion import apply_hidiffusion

    apply_hidiffusion(pipe)
    kwargs.update({"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5})
    return pipe, kwargs


@debug_monitor
def add_ays(pipe, kwargs, ays_type="StableDiffusionXLTimesteps"):
    """Apply AlignYourSteps optimization
    compatibility: stable-diffusion-xl, stable-diffusion, stable-video-diffusion
    """
    from diffusers.schedulers.scheduling_utils import AysSchedules

    # pipe, kwargs = solvers.dpmpp(pipe, kwargs, order=2)

    ays = AysSchedules[ays_type]
    kwargs.update(
        {
            "num_inference_steps": "10",
            "timesteps": ays,
        }
    )
    return pipe, kwargs


@debug_monitor
def add_generator(pipe, noise_seed: int = 0):
    """Create a generator object ready to receive seeds"""

    import torch

    pipe.generator = torch.Generator(pipe.device).manual_seed(noise_seed)
    return pipe


@debug_monitor
def dynamo_compile(pipe, unet: bool = True, vae: bool = True, transformer: bool = False):
    """
    Compile torch processes for speed
    """
    import torch

    if transformer:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    if unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    if vae:
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    return pipe


@debug_monitor
def get_func_name() -> str:
    """Return the name of the calling function for self-identification or diagnostic purposes"""
    from inspect import currentframe

    return currentframe().f_back.f_code.co_name
