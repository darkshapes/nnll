# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def run_inference(constructor_data: tuple,  prompts: dict, **user_set) -> None:
    """Dynamially build diffusion process based on model architecture\n
    :param constructor_data: Preassembled pipe
    :param prompt: Instructions to the generative model, defaults to ''
    """
    # :param user_set: List of LoRAs to add to the process, defaults to None
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

    prompt = prompts.get("text", "")
    pipe, model_id, pipe_call, generation = constructor_data
    nfo(f"pre-generator Model {model_id} Pipe {pipe} Arguments {generation}")
    generation.update(user_set)
    metadata = None
    content = None
    gen_data = {"parameters": {}}
    device = first_available()
    if "diffusers" in pipe_call[1].value[1].lower():
        pipe.to(device)
        pipe = add_generator(pipe=pipe, noise_seed=noise_seed)
        content = pipe(prompt=prompt, **generation).images[0]
        gen_data = disk.add_to_metadata(pipe=pipe, model=model_id, prompt=[prompt], kwargs=generation)
        # may also be video or audio!!
    elif "audiogen"in pipe_call[1].value[1].lower():
        pipe = next(iter(pipe))
        metadata = pipe.sample_rate
        pipe.to(device)
        content = pipe.generate([prompt])
        generation.update({"sample_rate": pipe.config.sampling_rate})
        gen_data = disk.add_to_metadata(pipe=pipe, model=model_id, prompt=[prompt], kwargs=generation)
    elif "parler_tts" in pipe_call[1].value[1].lower():
        input_ids = pipe[1](prompt).input_ids.to(device)
        prompt_input_ids = pipe[1](prompt).input_ids.to(device)
        pipe = pipe[0]
        pipe.to(device)
        generation = pipe.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        content = generation.cpu().numpy().squeeze()
        generation.update({"sampling_rate": pipe.config.sampling_rate})
        gen_data = disk.add_to_metadata(pipe=pipe, model=model_id, prompt=[prompt], kwargs=generation)
    if content:
        metadata = gen_data.get("parameters")
        nfo(f"content type output {content}, {type(content)}")
        disk.write_to_disk(content, metadata)
