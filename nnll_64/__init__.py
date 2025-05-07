### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
# import os
from typing import Any

from nnll_01 import debug_monitor


@debug_monitor
async def run_inference(mir_arch: str, lora_opt: list = None) -> Any:
    """Create diffusion process"""
    import nnll_59 as disk

    # from nnll_61 import HyperChain
    from nnll_62 import ConstructPipeline
    import nnll_56 as techniques
    from nnll_08 import soft_random, seed_planter
    from PIL import PngImagePlugin
    from nnll_16 import first_available

    noise_seed = soft_random()
    seed_planter(noise_seed)
    user_set = {  # needs to be abstracted out still
        "output_type": "pil",
        "noise_seed": noise_seed,
        "denoising_end": 1.5,
        "num_inference_steps": 30,
        "guidance_scale": 2.5,
        "eta": 1.0,
        "width": 768,
        "height": 1344,
        "safety_checker": False,
    }
    model_hash = {}
    active_gpu = first_available()

    prompt = "aquatic scene, sunken ship, ocean divers, coral, exotic fish"
    negative_prompt = ""
    lora = lora_opt
    # optimization = "ays"

    # data_chain = await HyperChain()
    factory = ConstructPipeline()
    pipe, model, kwargs = await factory.create_pipeline(architecture=mir_arch)

    if lora:
        pipe, model, kwargs = factory.add_lora(lora, "mir_arch", pipe)

    pipe.prompt = prompt
    if negative_prompt:
        pipe.prompt += negative_prompt
    pipe.to(active_gpu)

    pipe = await techniques.add_generator(pipe, noise_seed=user_set.get("noise_seed", 0))

    # generator
    kwargs.update(user_set)

    image = pipe(
        prompt=prompt,
        **kwargs,
    ).images[0]

    gen_data = await disk.add_to_metadata(pipe, prompt, model_hash, kwargs)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", str(gen_data.get("parameters")))

    data_chain.add_block(f"{pipe}{model}{kwargs}")
    await disk.write_image_to_disk(image, metadata)


### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# """Inferencing"""

# from PIL import PngImagePlugin

# from nnll_08 import soft_random, seed_planter
# import nnll_53 as mem
# from nnll_62 import ConstructPipeline
# import nnll_56 as techniques
# import nnll_57 as disk_op
# # import autoencoder as autoencoder

# pipelines = ConstructPipeline()

# noise_seed = soft_random()
# noise_seed_plot = seed_planter(noise_seed)
# user_set = {
#     "output_type": "pil",
#     "noise_seed": noise_seed,
#     "denoising_end": 1.5,
#     "num_inference_steps": 30,
#     "guidance_scale": 2.5,
#     "eta": 1.0,
#     "width": 768,
#     "height": 1344,
#     "safety_checker": False,
# }
# model = ""
# vae_file = ""
# lora_file = None
# active_gpu = "mps"

# prompt = "sunken ship, ocean divers"
# negative_prompt = ""
# architecture = "stable-diffusion-xl-base"
# # model 1
# pipe, model, kwargs = pipelines.create_pipeline(architecture)

# # model 2
# # vae_model = autoencoder.add_vae()
# # pipe.vae = vae_model

# # model 3
# pipe, lora, kwargs = pipelines.add_lora("spo", architecture, pipe, kwargs)
# # pipe, kwargs = techniques.add_slam(pipe, kwargs)

# # model 4
# # pipe, kwargs = solvers.euler_a(pipe, kwargs)


# pipe.prompt = prompt
# pipe.to(active_gpu)

# pipe = techniques.add_generator(pipe, noise_seed=user_set.get("noise_seed", 0))  # kolors, sdxl  # sigma, sdxl,

# # generator
# kwargs.update(user_set)
# mem.add_to_undo(**kwargs)
# gen_data = mem.add_to_metadata(pipe, model, prompt, kwargs)
# image = pipe(
#     prompt=prompt,
#     **kwargs,
# ).images[0]


# metadata = PngImagePlugin.PngInfo()
# metadata.add_text("parameters", str(gen_data.get("parameters")))


# disk_op.write_to_disk(image, metadata)

# # image_out: bool = True
# # model = look.sdxl_base_local
# # original_config = look.sdxl_base_config
# # pipe, kwargs = techniques.add_dmd(pipe, kwargs)
