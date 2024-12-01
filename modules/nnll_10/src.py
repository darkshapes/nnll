
import os
import sys

import torch
from diffusers import AutoencoderKL, AutoPipelineForText2Image
from diffusers.schedulers import AysSchedules
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
# from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from nnll_08.src import soft_random, seed_planter
from nnll_09.src import encode_prompt  # from nnll_18.src import get_pipeline_embeds
from nnll_23.src import DynamicMethodConstructor


# device = next(iter(set(supported_backends())))
queue = []
queue.extend([{
    "prompt": "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles",
    "seed": soft_random(),
}])

location = "your_pretrained_model_location"
expressions = {"some_key": "some_value"}

# encoder = method_crafter(nnll11.encoder_classes,"from_pretrained")
# clip_l = { "tokenizer_method_name": "from_pretrained", "location": "/Users/unauthorized/Downloads/models/metadata/CLI-VL", "local_files_only": True,  }
# clip_l = { "method_name": "from_pretrained", "location": "/Users/unauthorized/Downloads/models/metadata/CLI-VG", "local_files_only": True,}

class_dict = {
    "class_name": "CLIPTOKENIZER",
    "class_name": "CLIPTOKENIZERFAST",
    "class_name": "CLIPTEXTMODEL",
    "class_name": "CLIPTEXTMODELWITHPROJECTION",
}
model_expressions = {
    "use_safetensors": True,
    "local_files_only": True,
    "torch_dtype": torch.float16,
    "variant": "fp16"
}


def create_encodings(device, queue, paths):
    tokenizer_models, text_encoder_models = define_encoders(paths, device)
    with torch.no_grad():
        for generation in queue:
            generation['embeddings'] = encode_prompt(
                [generation['prompt'], generation['prompt']],
                tokenizer_models, text_encoder_models
            )
    return queue

    del tokenizer, text_encoder, tokenizer_2, text_encoder_2


encoders = define_encoders(device, tokenizer_dict, text_encoder_dict)
encodings = create_encodings(queue, clip, clip2, device)


vae_file = "/Users/unauthorized/Downloads/models/image/sdxl.vae.safetensors"
config_file = "/Users/unauthorized/Downloads/models/metadata/STA-XL/config.json"
vae = AutoencoderKL.from_single_file(vae_file, config=config_file, local_files_only=True, torch_dtype=torch.float16, variant="fp16").to(device)

model = "/Users/unauthorized/Downloads/models/metadata/STA-XL"

pipe = AutoPipelineForText2Image.from_pretrained(
    model,
    torch_dtype=torch.float16,
    variant="fp16",
    tokenizer=None,
    text_encoder=None,
    tokenizer_2=None,
    text_encoder_2=None,
    local_files_only=True,
    vae=vae
).to(device)

prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, "cuda")


ays = AysSchedules["StableDiffusionXLTimesteps"]

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")

generator = torch.Generator(device=device)

for i, generation in enumerate(queue, start=1):
    seed_planter(generation['seed'])
    generator.manual_seed(generation['seed'])

    generation['latents'] = pipe(
        prompt_embeds=generation['embeddings'][0],
        negative_prompt_embeds=generation['embeddings'][1],
        pooled_prompt_embeds=generation['embeddings'][2],
        negative_pooled_prompt_embeds=generation['embeddings'][3],
        num_inference_steps=10,
        timesteps=ays,
        guidance_scale=5,
        generator=generator,
        output_type='latent',
    ).images

image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]

image.save("done.png")


pipe.upcast_vae()
output_dir = "./"
with torch.no_grad():
    counter = [s.endswith('png') for s in output_dir].count(True)  # get existing images
    for i, generation in enumerate(queue, start=1):
        generation['latents'] = generation['latents'].to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(
            generation['latents'] / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        image = pipe.image_processor.postprocess(image, output_type='pil')[0]

        counter += 1
        filename = f"Shadowbox-{counter}-batch-{i}.png"

        image.save(os.path.join(output_dir, filename))  # optimize=True,
