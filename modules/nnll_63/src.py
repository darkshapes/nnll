# pylint: disable=import-outside-toplevel
# pylint:disable=line-too-long
import torch
import secrets
import random
from inspect import currentframe


def add_generator(pipe, noise_seed: int = 0):
    """Create a generator object ready to receive seeds"""
    pipe.generator = torch.Generator(pipe.device).manual_seed(noise_seed)
    return pipe


def soft_random(size: int = 0x2540BE3FF) -> int:
    """
    Generate a deterministic random number using philox\n
    :params size: `int` RNG ceiling in hex format
    :returns: `int` a random number of the specified length\n
    pair with `random.seed()` for best effect
    """
    try:
        from numpy.random import SeedSequence, Generator, Philox
    except ImportError as error_log:
        print(f"{error_log} numpy not installed.")
    else:
        entropy = f"0x{secrets.randbits(128):x}"  # good entropy
        rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    return int(rndmc.integers(0, size))


def seed_planter(seed, deterministic=True):
    """Drop seed into all possible locations"""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() is True:
        if deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available() is True:
        torch.mps.manual_seed(seed)
    elif torch.xpu.is_available() is True:
        torch.xpu.manual_seed(seed)


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

    # pipe, kwargs = solvers.ddim(pipe, kwargs)
    apply_hidiffusion(pipe)
    kwargs.update({"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5})
    return pipe, kwargs


def add_ays(pipe, kwargs, ays_type="StableDiffusionXLTimesteps"):
    """AlignYourSteps noise schedule (SD1, SDXL, SVD)"""

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
