# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

import os
import torch
from diffusers.utils import load_image
from PIL import PngImagePlugin

import modules.solvers
import modules.memory as mem
import modules.lookup as look
import modules.pipelines as pipelines
import modules.techniques as techniques
import modules.disk_op as disk_op
import modules.autoencoder as autoencoder

noise_seed = techniques.soft_random()
noise_seed_plot = techniques.seed_planter(noise_seed)
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
model = look.sdxl_base_local
vae_file = look.sdxl_base_vae
lora_file = None
active_gpu = "mps"

prompt = "sunken ship, ocean divers"
negative_prompt = ""

# model 1
pipe, kwargs = pipelines.sdxl_base_single_pipe(model=model)

# model 2
vae_model = autoencoder.add_vae()
pipe.vae = vae_model

# model 3
pipe, kwargs = techniques.add_ays(pipe, kwargs)
pipe, kwargs = techniques.add_slam(pipe, kwargs)

# model 4
# pipe, kwargs = solvers.euler_a(pipe, kwargs)


pipe.prompt = prompt
pipe.to(active_gpu)

pipe = techniques.add_generator(pipe, noise_seed=user_set.get("noise_seed", 0))  # kolors, sdxl  # sigma, sdxl,

# generator
kwargs.update(user_set)
mem.add_to_undo(kwargs)
gen_data = mem.add_to_metadata(pipe, model, prompt, kwargs)
image = pipe(
    prompt=prompt,
    **kwargs,
).images[0]


metadata = PngImagePlugin.PngInfo()
metadata.add_text("parameters", str(gen_data.get("parameters")))


disk_op.write_to_disk(image, metadata)

# image_out: bool = True
# model = look.sdxl_base_local
# original_config = look.sdxl_base_config
# pipe, kwargs = techniques.add_dmd(pipe, kwargs)
