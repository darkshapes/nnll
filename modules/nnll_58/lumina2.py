import torch
from diffusers import Lumina2Text2ImgPipeline
import modules.disk_op as disk_op
import modules.lookup as look
import set_dtype
import modules.inference as inference

pipe = Lumina2Text2ImgPipeline.from_pretrained(
    look.lumina2,
    torch_dtype=set_dtype.lumina2,
).to(set_dtype.device)

prompt = inference.prompt
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    # cfg_trunc_ratio=0.25,
    # cfg_normalization=True,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

disk_op.write_to_disk(image)
