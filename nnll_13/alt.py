# import requests
# import torch
# import os
# import io
# from PIL import Image
# import soundfile as sf
# from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from urllib.request import urlopen
# from package.scripts.device_check import device

# attn_type = "eager"
# if device == "cuda":
#     capability = "".join(list(str(x) for x in torch.cuda.get_device_capability()))
#     if int(capability) > 86:
#         attn_type = "flash_attention_2"


# # Define model path
# model_path = "microsoft/Phi-4-multimodal-instruct"

# # Load model and processor
# processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map=device,
#     torch_dtype="auto",
#     trust_remote_code=True,
#     # if you do not use Ampere or later GPUs, change attention to "eager"
#     _attn_implementation=attn_type,
# ).to(device)

# # Load generation config
# generation_config = GenerationConfig.from_pretrained(model_path)

# # Define prompt structure
# user_prompt = "<|user|>"
# assistant_prompt = "<|assistant|>"
# prompt_suffix = "<|end|>"

# # Part 1: Image Processing
# print("\n--- IMAGE PROCESSING ---")
# image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# prompt = f"{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}"
# print(f">>> Prompt\n{prompt}")

# # Download and open image
# image = Image.open(requests.get(image_url, stream=True, timeout=10).raw)
# inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# # Generate response
# generate_ids = model.generate(
#     **inputs,
#     max_new_tokens=1000,
#     generation_config=generation_config,
# )
# generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(f">>> Response\n{response}")

# # Part 2: Audio Processing
# print("\n--- AUDIO PROCESSING ---")
# audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
# speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
# prompt = f"{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}"
# print(f">>> Prompt\n{prompt}")

# # Download and open audio file
# audio, samplerate = sf.read(io.BytesIO(urlopen(audio_url).read()))

# # Process with the model
# inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors="pt").to(device)

# generate_ids = model.generate(
#     **inputs,
#     max_new_tokens=1000,
#     generation_config=generation_config,
# )
# generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(f">>> Response\n{response}")
