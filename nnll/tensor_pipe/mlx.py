# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from nnll.metadata.save_generation import write_to_disk


def generate_mlx_lm():
    from mlx_lm import load, generate
    import mlx.core as mx  # vlm
    from mlx_vlm import load, generate  # vlm
    from mlx_vlm.prompt_utils import apply_chat_template  # vlm
    from mlx_vlm.utils import load_config  # vlm

    model_path = ""
    model, processor = load(model_path)  # vl
    config = load_config(model_path)  # vlm_vision
    config = model.config  # vlm audio

    # image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]  # vlm
    # audio = ["/path/to/audio1.wav", "/path/to/audio2.mp3"]  # vlm audio
    # image = [Image.open("...")] can also be used with PIL.Image.Image objects
    prompt = ""

    messages = [{"role": "user", "content": prompt}]  # lm
    prompt = processor.apply_chat_template(
        messages,
    )  # lm
    formatted_prompt = apply_chat_template(
        prompt,
        # add_generation_prompt=True lm
        # processor, vlm
        # config, vlm
        # num_images=len(image), #vision
        # num_audios = len(audio) #audio
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        # image,
        # audio=audio,
        verbose=False,
    )
    print(output)


def generate_mflux():
    from mflux.flux.flux import Flux1
    from mflux.config.config import Config

    # Load the model
    flux = Flux1.from_name(
        model_name="schnell",  # "schnell" or "dev"
        quantize=8,  # 4 or 8
    )

    # Generate an image
    image = flux.generate_image(
        seed=2,
        prompt="Luxury food photograph",
        config=Config(
            num_inference_steps=2,  # "schnell" works well with 2-4 steps, "dev" works well with 20-25 steps
            height=1024,
            width=1024,
        ),
    )
    write_to_disk(content=image)


def generate_chroma():
    import sys
    import mlx.core as mx
    from chroma import ChromaPipeline
    import numpy as np
    import secrets
    import PIL

    def main(prompt):
        latent_size = (64, 64)

        chroma = ChromaPipeline(
            "chroma",
            download_hf=False,
            chroma_filepath="/Users/unauthorized/Downloads/models/chroma-unlocked-v46.safetensors",
            t5_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2",
            tokenizer_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2",
            vae_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        )

        size = 0x100000000
        entropy = f"0x{secrets.randbits(128):x}"  # good entropy
        rndmc = np.Generator(np.Philox(np.SeedSequence(int(entropy, 16))))
        seed = int(rndmc.integers(0, size))

        latent_generator = chroma.generate_latents(
            text=prompt,
            neg_text="",
            num_steps=28,
            seed=seed,
            latent_size=latent_size,
        )

        conditioning = next(latent_generator)
        (
            x_T,  # The initial noise
            x_positions,  # The integer positions used for image positional encoding
            t5_conditioning,  # The T5 features from the text prompt
            t5_positions,  # Integer positions for text (normally all 0s)
            neg_txt,
            neg_txt_ids,
        ) = conditioning

        mx.eval(conditioning)

        for x_t in latent_generator:
            mx.eval(x_t)

        latents = chroma.decode(x_t, latent_size=latent_size)

        denorm_latents = mx.clip((latents / 2 + 0.5), 0, 1)
        xpose_latents = mx.transpose(denorm_latents, (0, 2, 3, 1))
        cast_latents = mx.array.astype(xpose_latents, mx.float32)
        px_array = np.array(cast_latents)
        images = (px_array * 255).round().astype("uint8")
        images = [PIL.Image.fromarray(image) for image in images]
        for i in images:
            write_to_disk(content=images)


# if __name__ == "__main__":
#     main(sys.argv[0])
