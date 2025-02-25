### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint: disable=import-outside-toplevel
# pylint:disable=line-too-long
import torch
import secrets
import random
from inspect import currentframe


def ddim(pipe, kwargs, timestep="trailing", zero_snr=False):
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        timestep_spacing=timestep,  # compatibility for certain techniques
        subfolder="scheduler",
        rescale_betas_zero_snr=zero_snr,  # brighter and darker
    )
    pipe.scheduler = scheduler

    return pipe, kwargs


def dpmpp(pipe, kwargs, algorithm="dpmsolver++", order=2):
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        algorithm_type=algorithm,
        solver_order=order,
    )
    pipe.scheduler = scheduler

    return pipe, kwargs


def euler_a(pipe, kwargs):
    from diffusers import EulerAncestralDiscreteScheduler

    scheduler = EulerAncestralDiscreteScheduler()

    pipe.scheduler = scheduler

    return pipe, kwargs


def add_generator(pipe, noise_seed: int = 0):
    """Create a generator object ready to receive seeds"""
    pipe.generator = torch.Generator(pipe.device).manual_seed(noise_seed)
    return pipe


def get_func_name():
    """Return the name of the calling function for self-identification or diagnostic purposes"""
    return currentframe().f_back.f_code.co_name


def dynamo_compile(pipe, unet: bool = True, vae: bool = True, transformer: bool = False):
    """Compile components for speed"""
    if transformer:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    if unet:
        pipe.unload_lora_weights()
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    if vae:
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    return pipe


def add_hi_diffusion(pipe, kwargs):
    """Generate up to 2048 resolution\n
    (SD1, SD2, XL, Playground, Ghibli, I2I, ControlNet, Inpaint)"""
    from hidiffusion import apply_hidiffusion  # , remove_hidiffusion

    pipe, kwargs = ddim(pipe, kwargs)
    apply_hidiffusion(pipe)
    kwargs.update({"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5})
    return pipe, kwargs


def add_ays(pipe, kwargs, ays_type="StableDiffusionXLTimesteps"):
    """AlignYourSteps noise schedule (SD1, SDXL, SVD)"""

    from diffusers.schedulers.scheduling_utils import AysSchedules

    pipe, kwargs = dpmpp(pipe, kwargs, order=2)

    ays = AysSchedules[ays_type]
    kwargs.update(
        {
            "num_inference_steps": "10",
            "timesteps": ays,
        }
    )
    return pipe, kwargs
