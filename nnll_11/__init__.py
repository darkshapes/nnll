#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
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

    def __init__(self, sig: dspy.Signature = QASignature, max_workers=4) -> None:
        """
        Instantiate the module, setup parameters, create async streaming generator.\n
        Does not load any models until forward pass
        :param signature: The format of messages sent to the model
        :param max_workers: Maximum number of async processes, based on system resources
        """
        super().__init__()
        self.max_workers = max_workers
        generator = dspy.asyncify(program=dspy.Predict(signature=sig))  # this should only be used in the case of text
        self.completion = dspy.streamify(generator)

    # Reminder: Don't capture user prompts - this is the crucial stage
    async def forward(self, tx_data: dict[str | list[float]], model: str, library: LibType, streaming=True) -> Any:
        """
        Forward pass for LLM Chat process\n
        :param model: The library-specific arguments for the model configuration
        :param message: A simple string to send to the LLM
        :param tx_data: prompt transmission values for all media formats
        :param model: path to model
        :param library: LibType of model origin
        :param streaming: output type flag, defaults to True
        :yield: responses in chunks or response as a single block
        """

        from nnll_05 import lookup_function_for
        from httpx import ResponseNotRead

        nfo(f"libtype hub req : {model} {library}")
        if library == LibType.HUB:
            nfo(f"libtype hub req : {model}")
            constructor, mir_arch = await lookup_function_for(model)
            dbug(constructor, mir_arch)
            await constructor(mir_arch)

        else:
            try:
                api_kwargs = await get_api(model=model, library=library)
            except ValueError as error_log:
                nfo(f"Library '{library}' not found in configuration.")
                dbug(error_log)
                yield {
                    "choices": {
                        "0": {"delta": {"content": "The attempt to gather resources for this request was rejected. Have files changed?"}},
                    },
                }
            else:
                model = dspy.LM(**api_kwargs)
                dspy.settings.configure(lm=model, async_max_workers=self.max_workers)
                try:
                    yield self.completion(message=tx_data["text"], stream=streaming)
                except (GeneratorExit, RuntimeError, AttributeError, ResponseNotRead, ValueError) as error_log:
                    dbug(error_log)  # consider threading user selection between cursor jumps
                except TypeError as error_log:
                    dbug(error_log)
