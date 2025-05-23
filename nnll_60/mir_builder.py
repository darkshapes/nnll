### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from collections import defaultdict


def build_mir_db():
    """Create mir info database"""
    # from nnll_01 import nfo
    from nnll_07 import add_mir_entry
    from pprint import pprint

    entry = defaultdict(dict)
    element = add_mir_entry(
        domain="info",
        arch="unet",
        series="stable-diffusion-xl",
        compatibility="base",
        gen_kwargs={
            "num_inference_steps": 40,
            "denoising_end": 0.8,
            "output_type": "latent",
            "safety_checker": False,
        },
        init_kwargs={
            "use_safetensors": True,
        },
        dep_pkg=["diffusers"],
        layer_256=["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
        weight_map="weight_maps/model.unet.stable-diffusion-xl:base.json",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        module_path=["StableDiffusionXLPipeline"],
        alt_module=["DiffusionPipeline"],
        i2i_module=["StableDiffusionXLImg2ImgPipeline"],
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="info",
        arch="unet",
        series="stable-diffusion-xl",
        compatibility="refiner",
        repo="stabilityai/stable-diffusion-xl-refiner-1.0",
        layer_256=["8c2d0d32cff5a74786480bbaa932ee504bb140f97efdd1a3815f14a610cf6e4a"],
        weight_map="weight_maps/stable-diffusion-xl-refiner.json",
        module_path="DiffusionPipeline",
        dep_pkg=["diffusers"],
        gen_kwargs={"num_inference_steps": 40, "denoising_end": 0.8},
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="info",
        arch="unet",
        series="Kolors",
        compatibility="diffusers",
        repo="Kwai-Kolors/Kolors-diffusers",
        gen_kwargs={"negative_prompt": "", "guidance_scale": 5.0, "num_inference_steps": 50},
        pipe_kwargs={"torch_dtype": "torch.float16", "variant": "fp16"},
        pipe_name="KolorsPipeline",
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="ops",
        arch="scheduler",
        series="align-your-steps",
        compatibility="stable-diffusion-xl",
        num_inference_steps=10,
        timesteps="StableDiffusionXLTimesteps",
        dep_pkg="diffusers",
        module_path=["schedulers.scheduling_utils", "AysSchedules"],
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="ops",
        arch="patch",
        series="hidiffusion",
        compatibility="stable-diffusion-xl",
        num_inference_steps=10,
        timesteps="StableDiffusionXLTimesteps",
        dep_pkg=["hidiffusion"],
        repo="megvii-research/HiDiffusion/",
        gen_kwargs={"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5},
        module_path=["apply_hidiffusion"],
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="info",
        arch="unet",
        series="stable-cascade",
        compatibility="c",
        layer_256=[
            "2b6986954d9d2b0c702911504f78f5021843bd7050bb10444d70fa915cb495ea",
            "2aa5a461c4cd0e2079e81554081854a2fa01f9b876d7124c8fff9bf1308b9df7",
            "ce474fd5da12f1d465a9d236d61ea7e98458c1b9d58d35bb8412b2acb9594f08",
            "1b035ba92da6bec0a9542219d12376c0164f214f222955024c884e1ab08ec611",
            "22a49dc9d213d5caf712fbf755f30328bc2f4cbdc322bcef26dfcee82f02f147",
        ],
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    element = add_mir_entry(
        domain="info",
        arch="unet",
        series="stable-cascade",
        compatibility="b",
        layer_256=[
            "fde5a91a908e8cb969f97bcd20e852fb028cc039a19633b0e1559ae41edeb16f",
            "24fa8b55d12bf904878b7f2cda47c04c1a92da702fe149e28341686c080dfd4f",
            "a7c96afb54e60386b7d077bf3f00d04596f4b877d58e6a577f0e1a08dc4a0190",
            "f1300b9ffe051640555bfeee245813e440076ef90b669332a7f9fb35fffb93e8",
            "047fa405c9cd5ad054d8f8c8baa2294fbc663e4121828b22cb190f7057842a64",
        ],
    )
    entry[next(iter(element))].update(element[next(iter(element))])
    pprint(entry)


if __name__ == "__main__":
    build_mir_db()
