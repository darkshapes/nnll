### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def run_inference(mir_arch: str, tx_data: dict, out_type: str, lora_opt: list = None, **user_set) -> None:
    """Dynamially build diffusion process based on model architecture\n
    :param mir_arch: MIR system classifier string
    :param prompt: Instructions to the generative model, defaults to ''
    :param lora_opt: List of LoRAs to add to the process, defaults to None
    """
    from PIL import PngImagePlugin
    from nnll_01 import nfo  # , dbug
    from nnll_08 import soft_random, seed_planter
    from nnll_16 import first_available
    from nnll_62 import ConstructPipeline
    import nnll_56 as techniques
    import nnll_59 as disk

    user_set = {
        "output_type": "pil",
    }
    noise_seed = soft_random()
    seed_planter(noise_seed)
    nfo(noise_seed)

    # memory threshold formula function returns boolean value here
    factory = ConstructPipeline()
    pipe, model, kwargs = factory.create_pipeline(architecture=mir_arch, lora=lora_opt)
    nfo(f"pre-generator Model {model} Lora {lora_opt} Arguments {kwargs} {pipe}")

    pipe.to(first_available())

    prompt = tx_data.get("text", "")
    # if tx_data.get("image", 0):
    #     kwargs.setdefault("images", tx_data["image"])
    if out_type == "speech":
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
        kwargs.update(user_set)
        generate_ids = model.generate(**kwargs)
        generate_ids = generate_ids[:, kwargs["inputs"].shape[1] :]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        nfo(response)
    else:
        pipe = techniques.add_generator(pipe=pipe, noise_seed=noise_seed)
        kwargs.update(user_set)
        nfo(f"Pipe {pipe}, Device {pipe.device}")
        image = pipe(prompt=prompt, **kwargs).images[0]
        gen_data = disk.add_to_metadata(pipe=pipe, model=model, prompt=[prompt], kwargs=kwargs)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", str(gen_data.get("parameters")))
        disk.write_image_to_disk(image, metadata)

    # from nnll_61 import HyperChain
    # data_chain = HyperChain()
    # data_chain.add_block(f"{pipe}{model}{kwargs}")
