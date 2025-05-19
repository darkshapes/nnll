### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


def run_inference(mir_arch: str, tx_data: dict, lora_opt: list = None, **user_set) -> None:
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
    join = True  # memory threshold formula function returns boolean value here
    factory = ConstructPipeline()

    if join:
        pipe, model, kwargs = factory.create_pipeline(architecture=mir_arch, join=join)
    else:
        pipe_name, pipe_mode, pipe_kwargs, model, kwargs = factory.create_pipeline(architecture=mir_arch, join=join)  # pylint: disable=unbalanced-tuple-unpacking
        pipe_class = getattr(pipe_name, pipe_mode)
        pipe = pipe_class(model, **pipe_kwargs).to(first_available())

    if lora_opt:
        for lora in lora_opt:
            pipe, model, kwargs = factory.add_lora(lora, "mir_arch", pipe, kwargs)

    nfo(f"pre-generator Model {model} Lora {lora_opt} Arguments {kwargs} {pipe}")

    prompt = tx_data.get("text", "")
    if tx_data.get("image", 0):
        kwargs.setdefault("image", tx_data["image"])

    pipe = techniques.add_generator(pipe=pipe, noise_seed=noise_seed)
    if join:
        pipe.to(first_available())

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
