import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import io
from package.scripts.device_check import device

# --- Parler-TTS Configuration ---

parler_prompt = "Hey, how are you doing today? My name is Jenny, and I'm here to help you with any questions you have."
parler_description = "Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality."

multimodal_audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
multimodal_speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."




multimodal_audio, multimodal_samplerate = sf.read(io.BytesIO(urlopen(multimodal_audio_url).read()))

multimodal_inputs = multimodal_processor(text=multimodal_prompt, audios=[(multimodal_audio, multimodal_samplerate)], return_tensors="pt").to(device)

multimodal_generate_ids = multimodal_model.generate(
    **multimodal_inputs,
    max_new_tokens=1000,
    generation_config=multimodal_generation_config,
)
multimodal_generate_ids = multimodal_generate_ids[:, multimodal_inputs["input_ids"].shape[1] :]
multimodal_response = multimodal_processor.batch_decode(multimodal_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# --- Parler-TTS Generation ---
parler_generation = parler_model.generate(input_ids=parler_input_ids, prompt_input_ids=parler_prompt_input_ids)
parler_audio_arr = parler_generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", parler_audio_arr, parler_model.config.sampling_rate)

print(f">>> Multimodal Response:\n{multimodal_response}")