# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

# pylint: disable=import-outside-toplevel

# from diffusers import EulerDiscreteScheduler
# from diffusers import EulerAncestralDiscreteScheduler
# from diffusers import FlowMatchEulerDiscreteScheduler
# from diffusers import EDMDPMSolverMultistepScheduler
# from diffusers import HeunDiscreteScheduler
# from diffusers import UniPCMultistepScheduler
# from diffusers import LMSDiscreteScheduler
# from diffusers import DEISMultistepScheduler


def euler_a(pipe, kwargs):
    from diffusers import EulerAncestralDiscreteScheduler

    scheduler = EulerAncestralDiscreteScheduler()

    pipe.scheduler = scheduler

    return pipe, kwargs


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


def tcd(pipe, kwargs):
    from diffusers import TCDScheduler

    scheduler = TCDScheduler()
    kwargs.update(
        {
            "num_inference_steps": 4,
            "guidance_scale": 0,
            "eta": 0.3,
        }
    )
    pipe.scheduler = scheduler
    return pipe, kwargs


def lcm(pipe, kwargs):
    from diffusers import LCMScheduler

    scheduler = LCMScheduler()
    kwargs.update(
        {
            "num_inference_steps": 8,
        }
    )
    pipe.scheduler = scheduler
    return pipe, kwargs
