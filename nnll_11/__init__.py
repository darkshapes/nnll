#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=pointless-statement, unsubscriptable-object
import array
from typing import Any
import dspy
# from pydantic import BaseModel, Field

from nnll_01 import debug_monitor, dbug, nfo
from nnll_15.constants import LibType, has_api, LIBTYPE_CONFIG


ps_sysprompt = "Provide x for Y"
bqa_sysprompt = "Reply with short responses within 60-90 word/10k character code limits"
ps_infield_tag = "An image of x"
ps_outfield_tag = "The nature of the x in the image."
ps_edit_message = "Edited input image of the dog with a yellow hat."

is_msg: str = "Description x of the image to generate"
is_out: str = "An image matching the description x"


class I2ISignature(dspy.Signature):
    f"""{ps_sysprompt}"""
    # This is an example multimodal input signature
    image_input: dspy.Image = dspy.InputField(desc=ps_infield_tag)
    answer: str = dspy.OutputField(desc=ps_outfield_tag)
    image_output: dspy.Image = dspy.OutputField(desc=ps_edit_message)


class BasicImageSignature(dspy.Signature):
    message: str = dspy.InputField(desc=is_msg)
    image_output: dspy.Image = dspy.OutputField(desc=is_out)


class QASignature(dspy.Signature):
    f"""{bqa_sysprompt}"""

    message: str = dspy.InputField(desc="The message to respond to")
    # history: dspy.History = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 60 and 90 words and limited to 10000 character code blocks")


# signature: dspy.Signature = BasicQAHistory
@debug_monitor
async def get_api(model: str, library: LibType) -> dict:
    """
    Load model into chat completion method based on library and run query\n
    :param model: The model to create a reply with the question
    :param library: API Library to use
    :param _data: filled by config decorator, ignore, defaults to None
    :return: Arguments to pass to the LM constructor

    ====================================================
    #### IMPLIED
    Since model Libraries only populate registry if:
    - X : Library modules are detected at launch
    - Y : Library server was available on index

    THUS
    #### Safely assume only valid Libraries are processed
    `valid` in this case meaning available to the system out of the set of all Zodiac supported

    However, server status can change. This must be validated. So:

    #### GIVEN
    For any supported Library Type:
    - A: Library call modules MUST be detected in CONFIG data
    - B: Library server MUST be available
    - If A is True AND B is True: Library index operations will be run

    In theory a model can be removed while the server is rebooted in between these checks.\n
    We would have to repopulate and reconstruct the index to know. An expensive computation.\n
    It would be ideal to have a lookup method inside 'from_cache' that confirms\n
    - A : the model remains available in the library
    - B : the location of the model is real
    - C : the file exists
     Unfortunately, several local model API's do not have a method to determine model file location.

    Therefore:
    #### MODEL AVAILABILITY IS UNCERTAIN
    #### ALWAYS prepare a case where the model file itself cannot be found

    """

    @LIBTYPE_CONFIG.decorator
    def _read_data(data: dict = None):
        return data

    data = _read_data()
    req_form = {}

    if data.get(library.value[1], 0) and has_api(library.value[1]):
        config = data[library.value[1]]
        req_form = {
            "model": model,
            **config["api_kwargs"],
        }
        dbug("Pushing form : %s", req_form)
        return req_form
    raise ValueError(f"Library '{library}' not found in configuration.")


# Don't capture user prompts: AVOID logging this class as much as possible
class ChatMachineWithMemory(dspy.Module):
    """Base module for Q/A chats using async and `dspy.Predict` List-based memory
    Defaults to 5 question history, 4 max workers, and `HistorySignature` query"""

    model = None
    lora = None
    library = None
    sig: dspy.Signature = QASignature
    mir_arch = None
    mir_db = None
    pipe = None
    pipe_kwargs = None
    completion = None
    import_pkg = None

    def __init__(self, max_workers=4, streaming=True) -> None:
        """
        Instantiate the module, setup parameters, create async streaming generator.\n
        Does not load any models until forward pass
        :param signature: The format of messages sent to the model
        :param max_workers: Maximum number of async processes, based on system resources
        """
        from nnll_16 import first_available
        from nnll_60.mir_maid import MIRDatabase
        from nnll_62 import ConstructPipeline

        super().__init__()
        self.mir_db = MIRDatabase()
        self.factory = ConstructPipeline()
        self.device = first_available()
        self.max_workers = max_workers
        self.streaming = streaming

    async def prepare_model(self) -> None:
        """Load model in preparation of
        :param model: path to model
        :param library: LibType of model origin
        :param streaming: output type flag, defaults to True
        :yield: responses in chunks or response as a single block
        """
        from httpx import ResponseNotRead

        if self.library == LibType.HUB:
            # api_kwargs = await get_api(model=model, library=library)
            # generator = dspy.asyncify(constructor)
            # self.completion = dspy.streamify(generator)

            mir_arch = self.mir_db.find_path("repo", self.model.lower())
            series = self.mir_arch[0]
            arch_data = self.mir_db.database[series].get(mir_arch[1])
            init_modules = self.mir_db.database[series]["[init]"]
            self.pipe, self.model, self.import_pkg, self.pipe_kwargs = self.factory.create_pipeline(arch_data, init_modules)

            # lora=lora_opt)
            lora_arch = self.mir_db.database[series].get(self.lora[1])
            lora_repo = next(iter(lora_arch["repo"]))  # <- user location here OR this
            scheduler = self.mir_db.database[series]["[init]"].get("scheduler")
            kwargs = {}
            if scheduler:
                sched = self.mir_db.database[scheduler]["[init]"]
                scheduler_kwargs = self.mir_db.database[series]["[init]"].get("scheduler_kwargs")
                kwargs = {sched: sched, scheduler_kwargs: scheduler_kwargs}
            init_kwargs = lora_arch.get("init_kwargs")
            self.pipe = self.factory.add_lora(self.pipe, lora_repo=lora_repo, init_kwargs=init_kwargs, **kwargs)
        else:
            try:
                api_kwargs = await get_api(model=self.model, library=self.library)
            except (ValueError, ResponseNotRead) as error_log:
                nfo(f"Library '{self.library}' not found in configuration.")
                dbug(error_log)
                yield {
                    "choices": {
                        "0": {"delta": {"content": "Request attempt failed. Have file locations changed?"}},
                    },
                }
            else:
                if self.streaming:
                    generator = dspy.asyncify(program=dspy.Predict(signature=self.sig))  # this should only be used in the case of text
                    self.completion = dspy.streamify(generator)
                else:
                    self.completion = dspy.Predict(signature=self.sig)
                model = dspy.LM(**api_kwargs)
                dspy.settings.configure(lm=model, async_max_workers=self.max_workers)
                dbug(f"libtype hub req : {self.completion} {model} {self.library}")
        self.model = model

    # Reminder: Don't capture user prompts - this is the crucial stage
    async def forward(self, tx_data: dict[str | list[float]], out_type: str) -> Any:
        """
        Forward pass for multimodal process\n
        :param tx_data: prompt transmission values for all media formats
        :param streaming: output type flag, defaults to False
        """

        if self.library != LibType.HUB:
            #        history = dspy.History(messages=[{"question":tx_data["text"], "answer":last_answer}
            yield self.completion(message=tx_data["text"], stream=self.streaming)  # history=history)
        else:
            from nnll_16 import soft_random, seed_planter
            import nnll_56 as techniques
            import nnll_59 as disk

            noise_seed = seed_planter(soft_random())
            user_set = {
                "output_type": "pil",
            }
            # memory threshold formula function returns boolean value here
            prompt = tx_data.get("text", "")

            nfo(f"Pre-generator Model {self.model}  Pipe {self.pipe} Arguments {self.pipe_kwargs}")  # Lora {lora_opt}
            self.pipe_kwargs.update(user_set)
            metadata = None
            content = None
            gen_data = {"parameters": {}}

            if "diffusers" in self.import_pkg:
                self.pipe.to(self.device)
                self.pipe = techniques.add_generator(pipe=self.pipe, noise_seed=noise_seed)
                content = self.pipe(prompt=prompt, **self.pipe_kwargs).images[0]
                gen_data = disk.add_to_metadata(pipe=self.pipe, model=self.model, prompt=[prompt], kwargs=self.pipe_kwargs)
                # may also be video or audio!!

            elif "audiogen" in self.import_pkg:
                self.pipe = next(iter(self.pipe))
                metadata = self.pipe.sample_rate
                self.pipe.to(self.device)
                content = self.pipe.generate([prompt])
                self.pipe_kwargs.update({"sample_rate": self.pipe.config.sampling_rate})
                gen_data = disk.add_to_metadata(pipe=self.pipe, model=self.model, prompt=[prompt], kwargs=self.pipe_kwargs)

            elif "parler_tts" in self.import_pkg:
                input_ids = self.pipe[1](prompt).input_ids.to(self.device)
                prompt_input_ids = self.pipe[1](prompt).input_ids.to(self.device)
                self.pipe = self.pipe[0]
                self.pipe.to(self.device)
                generation = self.pipe.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                content = generation.cpu().numpy().squeeze()
                self.pipe_kwargs.update({"sampling_rate": self.pipe.config.sampling_rate})
                gen_data = disk.add_to_metadata(pipe=self.pipe, model=self.model, prompt=[prompt], kwargs=self.pipe_kwargs)

            if content:
                metadata = gen_data.get("parameters")
                nfo(f"content type output {content}, {type(content)}")
                disk.write_to_disk(content, metadata)

                # Uniqueness Tag
                # from nnll_61 import HyperChain
                # data_chain = HyperChain()
                # data_chain.add_block(f"{pipe}{model}{kwargs}")
