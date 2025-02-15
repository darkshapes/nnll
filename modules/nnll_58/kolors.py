import torch
import modules.disk_op as disk_op
from set_device import ACTIVE_GPU, TO_DTYPE, TO_PRECISION
from diffusers import KolorsPipeline
import modules.lookup as look
import set_dtype

pipe = KolorsPipeline.from_pretrained(look.kolors, set_dtype.kolors).to(set_dtype.device)

prompt = "當海盜招待女士們並在篝火旁唱歌時，一名忍者偷偷地下毒了一桶朗姆酒。"

image = pipe(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator(pipe.device).manual_seed(66),
).images[0]

disk_op.write_to_disk(image)
