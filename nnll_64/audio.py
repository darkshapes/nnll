def audiogen(repo, tx_data, kwargs):
    import torchaudio
    from audiocraft.models import AudioGen

    from audiocraft.data.audio import audio_write

    wav = pipe.generate(descriptions)

    for idx, one_wav in enumerate(wav):
        audio_write(f"{idx}", one_wav.cpu(), pipe.sample_rate, strategy="loudness", loudness_compressor=True)


def parler_tts(device):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    prompt = "Hey, how are you doing today?"
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."


def phi_4_tts(model, tx_data, kwargs):
    from transformers import AutoProcessor, GenerationConfig

    user = "<|user|>"
    audio_token = "<|audio_1|>"
    assistant = "<|assistant|>"
    suffix = "<|end|>"
    prompt = f"{user}{audio_token}{tx_data.get('text', '')}{suffix}{assistant}"
    processor = AutoProcessor.from_pretrained(model)
    kwargs.setdefault("inputs", processor(text=prompt, audios=tx_data["speech"], return_tensors="pt"))
    kwargs.setdefault("generation_config", GenerationConfig.from_pretrained(model, "generation_config.json"))
    kwargs.setdefault("max_new_tokens", 1200)
    generate_ids = model.generate(**kwargs)
    generate_ids = generate_ids[:, kwargs["inputs"].shape[1] :]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    nfo(response)
