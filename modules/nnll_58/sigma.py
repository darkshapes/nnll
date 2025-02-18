import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

import modules.disk_op as disk_op
import modules.inference as inference
import modules.lookup as look
import set_dtype

pipe = PixArtSigmaPipeline.from_pretrained(
    look.sigma,
    torch_dtype=set_dtype.sigma,
    use_safetensors=True,
)
pipe.to(set_dtype.device)


# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

prompt = inference.prompt
# prompt = "While the pirates are entertained by voluptuous ladies and sing by the bonfire, a ninja secretly poisons their barrel of rum."
image = pipe(prompt).images[0]

disk_op.write_to_disk(image)
