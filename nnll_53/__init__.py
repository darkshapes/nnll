### <!-- // /*  SPDX-License-Identifier: LAL-1.3) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from nnll_01 import debug_monitor
from nnll_44 import collect_hashes


@debug_monitor
def add_to_metadata(pipe, model, prompt, kwargs, negative_prompt=None):
    model_data = {}

    model_data.setdefault(collect_hashes(model))

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
