# # // SPDX-License-Identifier: blessing
# # // d a r k s h a p e s

# import os

from PIL import PngImagePlugin

import modules.nnll_59.src as disk
from modules.nnll_62.src import ConstructPipeline
import modules.nnll_63.src as techniques
from modules.nnll_61.src import HyperChain


def run_inference():
    noise_seed = techniques.soft_random()
    techniques.seed_planter(noise_seed)
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
    active_gpu = "mps"

    prompt = "aquatic scene, sunken ship, ocean divers, coral, exotic fish"
    negative_prompt = ""
    model = "lumina-2"
    # lora = "slam"
    # optimization = "ays"
    # factory.add_lora("", "", pipe)

    data_chain = HyperChain()
    factory = ConstructPipeline()
    pipe, model, kwargs = factory.create_pipeline("lumina-2")
    pipe.prompt = prompt
    pipe.to(active_gpu)
    pipe = techniques.add_generator(pipe, noise_seed=user_set.get("noise_seed", 0))

    # generator
    kwargs.update(user_set)

    image = pipe(
        prompt=prompt,
        **kwargs,
    ).images[0]

    gen_data = disk.form_metadata(pipe, prompt, model_hash, kwargs)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", str(gen_data.get("parameters")))

    data_chain.add_block(f"{pipe}{model}{kwargs}")
    disk.write_image_to_disk(image, metadata)
