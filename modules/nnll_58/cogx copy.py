import torch
import os
from diffusers import CogView3PlusPipeline, AutoencoderKL, DDIMScheduler

device = "mps"

pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B").to(device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)

queue = []d

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=False)

guidance_scale = 7.0
batch_size = 1
steps = 50
dimension_x = 1024
dimension_y = 1024
filename = "cogx3.png"

image = pipe(
    prompt=prompt,
    guidance_scale=guidance_scale,
    num_images_per_prompt=batch_size,
    num_inference_steps=steps,
    width=dimension_x,
    height=dimension_y,
    vae=vae,
).image[0]


pipe.upcast_vae()
image.save("test.png")
