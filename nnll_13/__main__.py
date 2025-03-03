import outetts

model_config = outetts.HFModelConfig_v2(model_path="OuteAI/OuteTTS-0.3-1B", tokenizer_path="OuteAI/OuteTTS-0.3-1B")
interface = outetts.InterfaceHF(model_version="0.3", cfg=model_config)

# interface.print_default_speakers()
# speaker = interface.load_default_speaker(name="en_female_1")

speaker = interface.create_speaker(audio_path="/Users/unauthorized/Documents/GitHub/darkshapes/nnll/nnll_13/c1.2.wav")
interface.save_speaker(speaker, "ts_speaker.json")
speaker = interface.load_speaker("ts_speaker.json")

gen_cfg = outetts.GenerationConfig(
    text="Viriel the orphan rapidly learned that bad behaviour was rewarded by attention from the high elf adults of his upbringing. His fosters, new to adoption and parenthood, never wanted the trouble or responsibility that came from such a child, and so the young elf was shuffled from home to home, finding no comfort of friends or familiarity. His playfully destructive conduct fused to the core of his personality as an irrevocable tenet, and when matched with equal parts resistance and rejection, being in the absence of his guardians was often better than the alternative.",
    temperature=0.4,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker,
)
output = interface.generate(config=gen_cfg)

output.save("output_test.wav")

# You can create a speaker profile for voice cloning, which is compatible across all backends.
# Load a default speaker
# Generate speech
# Save the generated speech to a file
# Print available default speakers
