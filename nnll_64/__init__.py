### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel
# import os
import copy
from typing import Any
# from nnll_01 import debug_monitor


# @debug_monitor
def run_inference(mir_arch: str, lora_opt: list = None) -> None:
    """Create diffusion process (temporary method)\n
    :param mir_arch: MIR system identifier
    :param lora_opt: List of LoRAs to add to the process, defaults to None\n
    """
    import nnll_59 as disk

    # from nnll_61 import HyperChain
    from nnll_62 import ConstructPipeline
    import nnll_56 as techniques
    from nnll_08 import soft_random, seed_planter
    from nnll_16 import first_available
    from PIL import PngImagePlugin
    from nnll_01 import nfo, dbug

    noise_seed = soft_random()
    seed_planter(noise_seed)
    nfo(noise_seed)

    # user_set = {  # needs to be abstracted out still
    #     "output_type": "pil",
    #     "num_inference_steps": 30,
    #     "guidance_scale": 2.5,
    #     "eta": 1.0,
    #     "width": 768,
    #     "height": 1344,
    # }

    user_set = {
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
    # import torch
    # import torch.multiprocessing as mp
    # mp.set_sharing_strategy("file_system")

    active_gpu = first_available()
    join = True  # memory threshold formula function returns boolean value here
    prompt = "aquatic scene, sunken ship, ocean divers, coral, exotic fish"
    negative_prompt = ""
    lora = lora_opt
    optimization = "ays"

    # data_chain = HyperChain()

    factory = ConstructPipeline()
    if join:
        pipe, model, kwargs = factory.create_pipeline(architecture=mir_arch, join=join)
    else:
        pipe_name, pipe_mode, pipe_kwargs, model, kwargs = factory.create_pipeline(architecture=mir_arch, join=join)
        pipe_class = getattr(pipe_name, pipe_mode)
        pipe = pipe_class(model, **pipe_kwargs).to(active_gpu)
    nfo(f"pre-generator Model {model} Lora {lora} Arguments {kwargs} {pipe}")
    if lora:
        pipe, model, kwargs = factory.add_lora(lora, "mir_arch", pipe, kwargs)

    pipe.prompt = prompt
    if negative_prompt:
        pipe.prompt += negative_prompt

    # pipe, lora, kwargs = factory.add_lora("spo", architecture, pipe, kwargs)
    # pipe, kwargs = techniques.add_slam(pipe, kwargs)

    pipe, kwargs = techniques.euler_a(pipe, kwargs)

    pipe = techniques.add_generator(pipe=pipe, noise_seed=noise_seed)

    pipe.to(active_gpu)
    kwargs.update(user_set)
    nfo(f"Pipe {pipe}, Device {pipe.device} - {f'Device {active_gpu}:0' == str(pipe.device)}")
    image = pipe(
        prompt=prompt,
        **kwargs,
    ).images[0]

    gen_data = disk.add_to_metadata(pipe, prompt, model_hash, kwargs)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", str(gen_data.get("parameters")))

    # data_chain.add_block(f"{pipe}{model}{kwargs}")
    disk.write_image_to_disk(image, metadata)


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
