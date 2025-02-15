import torch
from diffusers import CogView3PlusPipeline

import modules.disk_op as disk_op
import modules.inference as inference
import modules.lookup as look
import set_dtype


pipe = CogView3PlusPipeline.from_pretrained(
    look.cogview3,
    torch_dtype=set_dtype.cogview3,
).to(set_dtype.device)


prompt = inference.prompt
# prompt = "當海盜招待女士們並在篝火旁唱歌時，一名忍者偷偷地下毒了一桶朗姆酒。"

image = pipe(
    prompt=prompt,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

disk_op.write_to_disk(image)
