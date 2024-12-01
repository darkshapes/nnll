
import os
import sys
import torch
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer
from diffusers import FluxPipeline, AutoencoderKL


from nnll_09.src import encode_prompt
from nnll_14.src import supported_backends
from nnll_21.src import symlinker
from nnll_22.src import TextEncoderLink, UNetLink, AutoencoderLink, AbstractLink

device: str = next(iter(set(supported_backends())))

output_dir = "/Users/unauthorized/Pictures/output/"
prompt_path = os.path.join(output_dir, "prompt.txt")
with open(prompt_path, "r") as file:
    prompt = next(iter(line.strip() for line in file if line.strip() != ""), "")

encoder_a = TextEncoderLink()
encoder_b = TextEncoderLink()

clip_model = "/Users/unauthorized/Downloads/models/text/clip_l.flux1-dev.diffusers.safetensors"
t5_models = ["/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.1of2safetensors.safetensors", "/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.1of2safetensors.safetensors"]

encoder_1 = encoder_a.create_symlink(model_type="clip-l", target_path=clip_model)
encoder_2 = encoder_b.create_symlink(model_type="t5-xxl", target_path=t5_models)

tokenizer = CLIPTokenizer.from_pretrained(encoder_1)
tokenizer_2 = T5Tokenizer.from_pretrained(encoder_2)
text_encoder = CLIPTextModel.from_pretrained(encoder_1)
text_encoder_2 = T5EncoderModel.from_pretrained(encoder_2)


def create_encodings(device, prompt):
    with torch.no_grad():

        embeddings = encode_prompt(device,
                                   [prompt, prompt],
                                   [tokenizer, tokenizer_2], [text_encoder, text_encoder_2]
                                   )
    return embeddings


embeddings = create_encodings(device, prompt)

vae_file = "/Users/unauthorized/Downloads/models/image/neoptism_vae.diffusion_pytorch_model.safetensors"
config_file = "/Users/unauthorized/Downloads/models/metadata/flux1-dev/vae/config.json"
vae = AutoencoderKL.from_single_file(vae_file, config=config_file, local_files_only=True).to(device)

pipe = FluxPipeline.from_single_file(
    "/Users/unauthorized/Downloads/models/image/nepotismfux_v8Dit.safetensors",
    config="/Users/unauthorized/Downloads/models/metadata/flux1-dev",
    local_files_only=True,
    tokenizer=None,
    tokenizer_2=None,
    text_encoder=None,
    text_encoder_2=None,
    vae=vae,
    torch_dtype=torch.bfloat16
).to(device)

with torch.no_grad():
    image = pipe(
        prompt_embeds=embeddings[0],
        negative_prompt_embeds=embeddings[1],
        pooled_prompt_embeds=embeddings[2],
        negative_pooled_prompt_embeds=embeddings[3],
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator(device).manual_seed(0)
    ).images[0]

    file_prefix = "nnll_test"
    counter = [s.endswith('png') for s in output_dir].count(True)  # get existing images
    filename = f"{file_prefix}_{"{:02d}".format(counter)}.png"
    file_path = os.path.join(output_dir, filename)
    image.save(file_path)
