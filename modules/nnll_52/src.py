### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


"""Inferencing"""

from PIL import PngImagePlugin

from modules.nnll_08.src import soft_random, seed_planter
import modules.nnll_53.src as mem
from modules.nnll_62.src import ConstructPipeline
import modules.nnll_56.src as techniques
import modules.nnll_57.src as disk_op
# import modules.autoencoder as autoencoder

pipelines = ConstructPipeline()

noise_seed = soft_random()
noise_seed_plot = seed_planter(noise_seed)
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
model = ""
vae_file = ""
lora_file = None
active_gpu = "mps"

prompt = "sunken ship, ocean divers"
negative_prompt = ""
architecture = "stable-diffusion-xl-base"
# model 1
pipe, model, kwargs = pipelines.create_pipeline(architecture)

# model 2
# vae_model = autoencoder.add_vae()
# pipe.vae = vae_model

# model 3
pipe, lora, kwargs = pipelines.add_lora("spo", architecture, pipe, kwargs)
# pipe, kwargs = techniques.add_slam(pipe, kwargs)

# model 4
# pipe, kwargs = solvers.euler_a(pipe, kwargs)


pipe.prompt = prompt
pipe.to(active_gpu)

pipe = techniques.add_generator(pipe, noise_seed=user_set.get("noise_seed", 0))  # kolors, sdxl  # sigma, sdxl,

# generator
kwargs.update(user_set)
mem.add_to_undo(**kwargs)
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
