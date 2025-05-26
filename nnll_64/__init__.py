### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def run_inference(mir_arch: str, tx_data: dict, out_type: str, lora_opt: list = None, **user_set) -> None:
    """Dynamially build diffusion process based on model architecture\n
    :param mir_arch: MIR system classifier string
    :param prompt: Instructions to the generative model, defaults to ''
    :param lora_opt: List of LoRAs to add to the process, defaults to None
    """
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

    prompt = tx_data.get("text", "")
    factory = ConstructPipeline()
    pipe, model, import_pkg, kwargs = factory.create_pipeline(architecture=mir_arch, lora=lora_opt)
    nfo(f"pre-generator Model {model} Lora {lora_opt} Pipe {pipe} Arguments {kwargs}")
    kwargs.update(user_set)
    metadata = None
    save_as = None
    content = None
    device = first_available()
    if "diffusers" in import_pkg:
        pipe.to(device)
        pipe = techniques.add_generator(pipe=pipe, noise_seed=noise_seed)
        content = pipe(prompt=prompt, **kwargs).images[0]
        gen_data = disk.add_to_metadata(pipe=pipe, model=model, prompt=[prompt], kwargs=kwargs)
        save_as = ".png"  # may also be video or audio!!
        metadata = gen_data.get("parameters")
    elif "audiogen" in import_pkg:
        pipe = next(iter(pipe))
        metadata = pipe.sample_rate
        save_as = ".wav"
        pipe.to(device)
        content = pipe.generate([prompt])
    elif "parler_tts" in import_pkg:
        input_ids = pipe[1](prompt).input_ids.to(device)
        prompt_input_ids = pipe[1](prompt).input_ids.to(device)
        save_as = ".wav"
        pipe = pipe[0]
        pipe.to(device)
        generation = pipe.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        content = generation.cpu().numpy().squeeze()
        metadata = pipe.config.sampling_rate

    if content and save_as:
        nfo(f"content type output {content}, {type(content)}")
        disk.write_to_disk(content, metadata, save_as)

    # from nnll_61 import HyperChain
    # data_chain = HyperChain()
    # data_chain.add_block(f"{pipe}{model}{kwargs}")
