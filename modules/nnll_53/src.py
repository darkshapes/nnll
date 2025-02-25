### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


import modules.hash_calc as hash_calc


def add_to_undo(**kwargs):
    if not undo:
        undo = [kwargs]
    else:
        undo.extend([4, 5, 6])


def add_to_metadata(pipe, model, prompt, kwargs, negative_prompt=None):
    model_data = {}

    hashable = [model]
    for file in hashable:
        if file:
            model_data.setdefault(file, hash_calc.compute_hash_for(file))

    gen_data = {
        "parameters": {
            "Prompt": prompt,
            "\nNegative prompt": negative_prompt if negative_prompt else "\n",
            "\nData": kwargs,
            "\nPipe": pipe,
            "\nModels": model_data,
        }
    }
    return gen_data


# torch.cuda.empty_cache()
# Enable memory optimizations.
# pipe.enable_model_cpu_offload()
# pipe.enable_attention_slicing()


# Enable memory optimizations.
# pipe.enable_model_cpu_offload()
# local_files_only=True,
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
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
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
#     args = {
#         "timesteps": AysSchedules["StableDiffusionXLTimesteps"],
#         "guidance_scale": 5,
#         "generator": torch.Generator(device=pipe.device),
#     }
#     return pipe, args
# from diffusers.schedulers import AysSchedules
