# import torch
# from diffusers import DiffusionPipeline
# import disk_op
# import inference
# import lookup as look
# import set_dtype

# # load both base & refiner
# base = DiffusionPipeline.from_pretrained(
#     look.sdxl_base,
#     torch_dtype=set_dtype.sdxl_base,
#     variant=set_dtype.variant_sdxl_base,
#     use_safetensors=True,
# )

# base.to("mps")
# base.transformer = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

# refiner = DiffusionPipeline.from_pretrained(
#     look.sdxl_refiner,
#     text_encoder_2=base.text_encoder_2,
#     use_safetensors=True,
#     torch_dtype=set_dtype.sdxl_refiner,
#     variant=set_dtype.variant_sdxl_refiner,
# )
# refiner.to("mps")
# refiner.transformer = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
# refiner.transformer = torch.compile(refiner.vae, mode="reduce-overhead", fullgraph=True)

# # Define how many steps and what % of steps to be run on each experts (80/20) here

# prompt = inference.prompt

# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images

# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]

# disk_op.write_to_disk(image)
