# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def run_inference(pipe_data: tuple, prompts: dict, out_type: str, **user_set) -> None:
    """Dynamially build diffusion process based on model architecture\n
    :param constructor_data: Preassembled pipe
    :param prompt: Instructions to the generative model, defaults to ''
    """
    # :param user_set: List of LoRAs to add to the process, defaults to None
    from nnll.configure.chip_stats import ChipStats
    from nnll.configure.init_gpu import soft_random, seed_planter, first_available
    from nnll.tensor_pipe.segments import add_generator
    from nnll.metadata import save_generation as disk
    from nnll.monitor.console import nfo

    user_set = {
        "output_type": "pil",
    }
    noise_seed = soft_random()
    seed_planter(noise_seed)
    nfo(noise_seed)
    # memory threshold formula function returns boolean value here
    chip_stats = ChipStats()
    metrics = chip_stats.get_metrics()
    prompt = prompts.get("text", "")
    pipe, model, pipe_call, generation = pipe_data
    nfo(f"pre-generator Model {model} Pipe {pipe} Arguments {generation}")
    generation.update(user_set)
    metadata = None
    content = None
    device = first_available()
    pkg_name = pipe_call.get("pkg_name").lower()
    if "diffusers" in pkg_name:
        if out_type == "image":
            pipe.to(device)
            pipe = add_generator(pipe=pipe, noise_seed=noise_seed)
            content = pipe(prompt=prompt, **generation).images[0]
            file_type = ".png"
        if out_type == "video":
            pipe.to(device)
            pipe.vae.enable_tiling()
            pipe = add_generator(pipe=pipe, noise_seed=noise_seed)

            content = pipe(prompt=prompt, **generation).frames[0]
            file_type = ".mp4"
            # may also be video or audio!!
    elif "mflux" in pkg_name:
        from mflux.config.config import Config

        content = pipe.generate_image(seed=noise_seed, prompt=prompt, config=Config(**generation))
    elif "audiogen" in pkg_name:
        pipe = next(iter(pipe))
        metadata = pipe.sample_rate
        pipe.to(device)
        content = pipe.generate([prompt])
        generation.update({"sample_rate": pipe.config.sampling_rate})
        file_type = ".wav"
    elif "parler_tts" in pkg_name:
        input_ids = pipe[1](prompt).input_ids.to(device)
        prompt_input_ids = pipe[1](prompt).input_ids.to(device)
        pipe = pipe[0]
        pipe.to(device)
        generation = pipe.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        content = generation.cpu().numpy().squeeze()
        generation.update({"sampling_rate": pipe.config.sampling_rate})
        file_type = ".wav"
    if content:
        # from nnll.integrity.hashing import collect_hashescollect_hashes(kwargs["model"]
        metadata = {"parameters": {"pipe": pipe, "model": model, "prompt": prompts, "kwargs": generation}}
        nfo(f"content type output {content}, {type(content)}")
        disk.write_to_disk(content=content, metadata=metadata, extension=file_type)
