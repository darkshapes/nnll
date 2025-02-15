# pylint: disable=import-outside-toplevel
import torch
import modules.nnll_56.look as look
import solvers
import secrets
import random


def adapt_and_fuse(
    pipe,
    kwargs,
    terms: list[str] = None,
    weights: list[float] = None,
    scale: float = 1.0,
):
    fuse_args = {"lora_scale": scale}
    if terms:
        fuse_args.setdefault("adapter_names", terms)
        pipe.set_adapters(terms, adapter_weights=weights)
    pipe.fuse_lora(fuse_args)
    return pipe, kwargs


def add_spo(pipe, kwargs):
    pipe.load_lora_weights(look.spo_lora)
    kwargs.update({"guidance_scale": 5.0})
    return pipe, kwargs


def add_dmd(pipe, kwargs):
    pipe, kwargs = solvers.lcm(pipe, kwargs)

    pipe.load_lora_weights(look.dmd_lora)
    kwargs.update(
        {
            "num_inference_steps": 4,
            "guidance_scale": 0,
            "timesteps": [999, 749, 499, 249],
        }
    )
    return pipe, kwargs


def add_tcd(pipe, kwargs):
    pipe, kwargs = solvers.tcd(pipe, kwargs)

    pipe.load_lora_weights(look.tcd_lora)
    kwargs.update(
        {
            "num_inference_steps": 4,
            "guidance_scale": 0,
            "eta": 0.3,
        }
    )
    return pipe, kwargs


def add_slam(pipe, kwargs):
    pipe, kwargs = solvers.lcm(pipe, kwargs)

    pipe.load_lora_weights(look.slam_lora)
    kwargs.update(
        {
            "num_inference_steps": 4,
            "guidance_scale": 1,
        }
    )
    return pipe, kwargs


def add_hi_diffusion(pipe, kwargs):
    from hidiffusion import apply_hidiffusion, remove_hidiffusion

    apply_hidiffusion(pipe)
    kwargs.update({"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5})
    return pipe, kwargs


def add_ays(pipe, kwargs, ays_type="StableDiffusionXLTimesteps"):
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


def add_generator(pipe, noise_seed: int = 0):
    """Create a generator object ready to receive seeds"""
    pipe.generator = torch.Generator(pipe.device).manual_seed(noise_seed)
    return pipe


def dynamo_compile(pipe, unet: bool = True, vae: bool = True, transformer: bool = False):
    if transformer:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    if unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    if vae:
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
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
        ImportError(f"{error_log} numpy not installed.")
    else:
        entropy = f"0x{secrets.randbits(128):x}"  # good entropy
        rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
        return int(rndmc.integers(0, size))


def seed_planter(seed, deterministic=True):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() is True:
        if deterministic is True:
            return {"torch.backends.cudnn.deterministic": "True", "torch.backends.cudnn.benchmark": "False"}
        return torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available() is True:
        return torch.mps.manual_seed(seed)
    elif torch.xpu.is_available() is True:
        return torch.xpu.manual_seed(seed)
