#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=pointless-statement, unsubscriptable-object
from typing import Any
import dspy
from pydantic import BaseModel, Field

from nnll_01 import debug_monitor, debug_message as dbug  # , info_message as nfo
from nnll_15.constants import LibType, has_api, LibType, LIBTYPE_CONFIG


ps_sysprompt = "Provide x for Y"
bqa_sysprompt = "Reply with short responses within 60-90 word/10k character code limits"
ps_infield_tag = "An image of x"
ps_outfield_tag = "The nature of the x in the image."
ps_edit_message = "Edited input image of the dog with a yellow hat."


class PictureSignature(dspy.Signature):
    f"""{ps_sysprompt}"""
    image_input: dspy.Image = dspy.InputField(desc=ps_infield_tag)
    answer: str = dspy.OutputField(desc=ps_outfield_tag)
    image_output: dspy.Image = dspy.OutputField(desc=ps_edit_message)


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
    """

    @LIBTYPE_CONFIG.decorator
    def _read_data(data:dict =None):
        return data

    libtype_data = _read_data()
    # import requests
    # import importlib
    req_form = {}
    if library == LibType.OLLAMA:
        model = {"model": model, "api_base": "http://localhost:11434/api/chat", "model_type": "chat"}  # ollama_chat/
    elif library == LibType.LM_STUDIO and has_api("LM_STUDIO"):
        req_form = {"model": model, **libtype_data["LM_STUDIO"].get("api_kwargs")}  # lm_studio/
    elif library == LibType.HUB and has_api("HUB"):
        req_form = {"model": model, **libtype_data["HUB"].get("api_kwargs")}# huggingface/civitai via sdbx
    elif library == LibType.VLLM and has_api("VLLM"):
        req_form = {"model": model, **libtype_data["VLLM"].get("api_kwargs")} # hosted_vllm/
    elif library == LibType.LLAMAFILE and has_api("LLAMAFILE"):
        req_form = {"model": model, **libtype_data["LLAMAFILE"].get("api_kwargs")}
    elif library == LibType.CORTEX and has_api("CORTEX"):
        req_form = {"model": model, **libtype_data["CORTEX"].get("api_kwargs")}
    return model


# fact_checking = dspy.ChainOfThought('claims -> verdicts: list[bool]')
# fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."])


class ChatMachineWithMemory(dspy.Module):
    """Base module for Q/A chats using async and `dspy.Predict` List-based memory
    Defaults to 5 question history, 4 max workers, and `HistorySignature` query"""

    @debug_monitor
    def __init__(self, sig: dspy.Signature = QASignature, streaming=True) -> None:
        """
        Instantiate the module, setup parameters, create async streaming generator.\n
        Does not load any models until forward pass
        :param signature: The format of messages sent to the model
        """
        super().__init__()
        self.streaming = streaming
        generator = dspy.asyncify(program=dspy.Predict(signature=sig))  # this should only be used in the case of text
        self.completion = dspy.streamify(generator)

    @debug_monitor
    async def forward(self, tx_data: str, model: str, library: LibType, max_workers=4) -> Any:
        """
        Forward pass for LLM Chat process\n
        :param model: The library-specific arguments for the model configuration
        :param message: A simple string to send to the LLM
        :param max_workers: Maximum number of async processes
        :return: yields response in chunks
        """
        from nnll_05 import lookup_function_for
        from httpx import ResponseNotRead

        if library == LibType.HUB:
            constructor = await lookup_function_for(model)
            async for chunk in constructor(model):
                yield chunk
        else:
            api_kwargs = await get_api(model=model, library=library)
            model = dspy.LM(**api_kwargs)
            dspy.settings.configure(lm=model, async_max_workers=max_workers)
            async for chunk in self.completion(message=tx_data["text"], stream=self.streaming):
                try:
                    if chunk is not None:
                        if isinstance(chunk, dspy.Prediction):
                            if not self.streaming:
                                yield chunk.answer  # the final, processed output
                            else:
                                pass
                        else:
                            yield chunk["choices"][0]["delta"]["content"]
                except (GeneratorExit, RuntimeError, AttributeError, ResponseNotRead) as error_log:
                    dbug(error_log)  # consider threading user selection between cursor jumps
