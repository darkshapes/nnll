from transformers import (
    CLIPTokenizer,
    CLIPTokenizerFast,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)
from diffusers import (
    Transformer2DModel,
    Transformer2DModel,
)
from diffusers import AutoencoderKL
import modules.lookup as look


def add_vae(model=look.sdxl_base_vae):
    """
    Create variable autoencoder component\n
    """
    return (
        AutoencoderKL.from_pretrained(
            model,
            # torch_dtype=set_dtype.sdxl_base,
            # variant=set_dtype.variant_sdxl_base,
        ),
        model,
    )


# torch.cuda.empty_cache()
# Enable memory optimizations.
# pipe.enable_model_cpu_offload()
# pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_tiling()

# Enable memory optimizations.
# local_files_only=True,
# AutoPipelineForText2Image,
# DPMSolverMultistepScheduler,
# AutoencoderKL,
# EulerAncestralDiscreteScheduler


# def sdxl_pipe():
#     model = model
#     vae_file = ""
#     config_file = ""
#     pipe = AutoPipelineForText2Image.from_pretrained(
#         model,
#         torch_dtype=torch.float16,
#         variant="fp16",
#         tokenizer=None,
#         text_encoder=None,
#         tokenizer_2=None,
#         text_encoder_2=None,
#         local_files_only=True,
#         vae=vae,  # "C:\\Users\\Public\\models\\image\\flatpiecexlVAE_baseonA1579.safetensors"
#     ).to(ACTIVE_GPU)
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     #pipe.scheduler.config, algorithm_type="dpmsolver++")
#     args = {
#         "timesteps": AysSchedules["StableDiffusionXLTimesteps"],
#         "guidance_scale": 5,
#         "generator": torch.Generator(device=pipe.device),
#     }
#     return pipe, args
# from diffusers.schedulers import AysSchedules

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

clip = "C:\\Users\\Public\\models\\metadata\\CLI-VL"
clip2 = "C:\\Users\\Public\\models\\metadata\\CLI-VG"
from sdbx.nodes.helpers import soft_random, seed_planter
from sdbx.config import config


def encode_prompt(prompts, tokenizers, text_encoders):
    embeddings_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        cond_input = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        prompt_embeds = text_encoder(cond_input.input_ids.to("cuda"), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        embeddings_list.append(prompt_embeds.hidden_states[-2])

        prompt_embeds = torch.concat(embeddings_list, dim=-1)

    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(1 * 1, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, 1).view(bs_embed * 1, -1)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


tokenizer = CLIPTokenizer.from_pretrained(
    clip,
    local_files_only=True,
)

text_encoder = CLIPTextModel.from_pretrained(
    clip,
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
).to(device)

tokenizer_2 = CLIPTokenizer.from_pretrained(
    clip2,
    local_files_only=True,
)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    clip2,
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
).to(device)

with torch.no_grad():
    for generation in queue:
        generation["embeddings"] = encode_prompt(
            [generation["prompt"], generation["prompt"]],
            [tokenizer, tokenizer_2],
            [text_encoder, text_encoder_2],
        )

del tokenizer, text_encoder, tokenizer_2, text_encoder_2


max_memory = round(torch.cuda.max_memory_allocated(device=device) / 1e9, 2)
print("Max. memory used:", max_memory, "GB")

queue = []
prompt = "A slice of a rich and delicious chocolate cake presented on a table in a luxurious palace reminiscent of Versailles"
# seed = soft_random()
queue.extend(
    [
        {
            "prompt": prompt,
            # "seed": seed,
        }
    ]
)
